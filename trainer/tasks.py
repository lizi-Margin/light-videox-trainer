from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from models.wan_forward import encode_video_latents


@dataclass
class TaskCondition:
    y: torch.Tensor | None
    clip_fea: torch.Tensor | None
    loss_mask: torch.Tensor | None = None


class WanTrainingTask(ABC):
    name = "base"

    def __init__(self, vae_mini_batch: int = 1) -> None:
        self.vae_mini_batch = vae_mini_batch

    @abstractmethod
    def build_condition(self, bundle, pixel_values: torch.Tensor, latents: torch.Tensor) -> TaskCondition:
        raise NotImplementedError


class WanT2VTask(WanTrainingTask):
    name = "t2v"

    def build_condition(self, bundle, pixel_values: torch.Tensor, latents: torch.Tensor) -> TaskCondition:
        transformer = bundle.unwrap_transformer()
        if getattr(transformer, "model_type", "t2v") != "i2v":
            return TaskCondition(y=None, clip_fea=None)

        in_channels = int(transformer.config.in_channels)
        condition_channels = in_channels - int(latents.shape[1])
        if condition_channels <= 0:
            return TaskCondition(y=None, clip_fea=None)

        y = latents.new_zeros(
            latents.shape[0],
            condition_channels,
            latents.shape[2],
            latents.shape[3],
            latents.shape[4],
        )
        clip_fea = latents.new_zeros(latents.shape[0], 257, 1280)
        return TaskCondition(y=y, clip_fea=clip_fea)


class InpaintMaskGenerator:
    def __init__(
        self,
        modes: list[str] | None = None,
        full_frame_ratio: float = 0.2,
        temporal_static_ratio: float = 0.5,
        min_area_ratio: float = 0.08,
        max_area_ratio: float = 0.55,
        brush_strokes: tuple[int, int] = (4, 12),
        brush_width: tuple[int, int] = (24, 96),
    ) -> None:
        self.modes = modes or ["rectangle", "moving_rectangle", "brush"]
        self.full_frame_ratio = full_frame_ratio
        self.temporal_static_ratio = temporal_static_ratio
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.brush_strokes = brush_strokes
        self.brush_width = brush_width

    @classmethod
    def from_config(cls, cfg: dict[str, Any], prefix: str = "mask") -> "InpaintMaskGenerator":
        modes = cfg.get(f"{prefix}_modes", cfg.get("mask_modes"))
        if isinstance(modes, str):
            modes = [m.strip() for m in modes.split(",") if m.strip()]
        return cls(
            modes=modes,
            full_frame_ratio=float(cfg.get(f"{prefix}_full_frame_ratio", cfg.get("mask_full_frame_ratio", 0.2))),
            temporal_static_ratio=float(cfg.get(f"{prefix}_temporal_static_ratio", cfg.get("mask_temporal_static_ratio", 0.5))),
            min_area_ratio=float(cfg.get(f"{prefix}_min_area_ratio", cfg.get("mask_min_area_ratio", 0.08))),
            max_area_ratio=float(cfg.get(f"{prefix}_max_area_ratio", cfg.get("mask_max_area_ratio", 0.55))),
            brush_strokes=(
                int(cfg.get(f"{prefix}_brush_strokes_min", cfg.get("mask_brush_strokes_min", 4))),
                int(cfg.get(f"{prefix}_brush_strokes_max", cfg.get("mask_brush_strokes_max", 12))),
            ),
            brush_width=(
                int(cfg.get(f"{prefix}_brush_width_min", cfg.get("mask_brush_width_min", 24))),
                int(cfg.get(f"{prefix}_brush_width_max", cfg.get("mask_brush_width_max", 96))),
            ),
        )

    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, frames, _, height, width = pixel_values.shape
        mask = pixel_values.new_zeros(batch, frames, 1, height, width)
        for idx in range(batch):
            if torch.rand((), device=pixel_values.device) < self.full_frame_ratio:
                mask[idx] = 1
                continue
            mode = self.modes[int(torch.randint(0, len(self.modes), (), device=pixel_values.device).item())]
            if mode == "moving_rectangle":
                self._moving_rectangle(mask[idx], frames, height, width, pixel_values.device)
            elif mode == "brush":
                self._brush(mask[idx], frames, height, width, pixel_values.device)
            else:
                self._rectangle(mask[idx], frames, height, width, pixel_values.device)
        return mask

    def _rectangle(self, mask: torch.Tensor, frames: int, height: int, width: int, device: torch.device) -> None:
        block_w, block_h = self._block_size(height, width, device)
        left = int(torch.randint(0, max(1, width - block_w + 1), (), device=device).item())
        top = int(torch.randint(0, max(1, height - block_h + 1), (), device=device).item())
        start_f, end_f = self._frame_range(frames, device)
        mask[start_f:end_f, :, top : top + block_h, left : left + block_w] = 1

    def _moving_rectangle(self, mask: torch.Tensor, frames: int, height: int, width: int, device: torch.device) -> None:
        block_w, block_h = self._block_size(height, width, device)
        start_left = int(torch.randint(0, max(1, width - block_w + 1), (), device=device).item())
        start_top = int(torch.randint(0, max(1, height - block_h + 1), (), device=device).item())
        end_left = int(torch.randint(0, max(1, width - block_w + 1), (), device=device).item())
        end_top = int(torch.randint(0, max(1, height - block_h + 1), (), device=device).item())
        start_f, end_f = self._frame_range(frames, device)
        denom = max(1, end_f - start_f - 1)
        for frame in range(start_f, end_f):
            alpha = (frame - start_f) / denom
            left = int(round(start_left * (1.0 - alpha) + end_left * alpha))
            top = int(round(start_top * (1.0 - alpha) + end_top * alpha))
            mask[frame, :, top : top + block_h, left : left + block_w] = 1

    def _brush(self, mask: torch.Tensor, frames: int, height: int, width: int, device: torch.device) -> None:
        start_f, end_f = self._frame_range(frames, device)
        strokes = int(torch.randint(self.brush_strokes[0], self.brush_strokes[1] + 1, (), device=device).item())
        for _ in range(strokes):
            stroke_w = int(torch.randint(self.brush_width[0], self.brush_width[1] + 1, (), device=device).item())
            points = int(torch.randint(2, 6, (), device=device).item())
            ys = torch.randint(0, height, (points,), device=device)
            xs = torch.randint(0, width, (points,), device=device)
            for point_idx in range(points - 1):
                self._draw_line(mask[start_f:end_f], xs[point_idx], ys[point_idx], xs[point_idx + 1], ys[point_idx + 1], stroke_w)

    def _draw_line(self, mask: torch.Tensor, x0: torch.Tensor, y0: torch.Tensor, x1: torch.Tensor, y1: torch.Tensor, width: int) -> None:
        height, image_width = mask.shape[-2:]
        steps = int(max(abs(int(x1.item()) - int(x0.item())), abs(int(y1.item()) - int(y0.item())), 1))
        xs = torch.linspace(float(x0.item()), float(x1.item()), steps, device=mask.device).round().long()
        ys = torch.linspace(float(y0.item()), float(y1.item()), steps, device=mask.device).round().long()
        radius = max(1, width // 2)
        for x, y in zip(xs, ys):
            left = max(0, int(x.item()) - radius)
            right = min(image_width, int(x.item()) + radius)
            top = max(0, int(y.item()) - radius)
            bottom = min(height, int(y.item()) + radius)
            mask[:, :, top:bottom, left:right] = 1

    def _block_size(self, height: int, width: int, device: torch.device) -> tuple[int, int]:
        min_scale = self.min_area_ratio**0.5
        max_scale = self.max_area_ratio**0.5
        w_scale = torch.empty((), device=device).uniform_(min_scale, max_scale).item()
        h_scale = torch.empty((), device=device).uniform_(min_scale, max_scale).item()
        block_w = min(width, max(1, int(round(width * w_scale))))
        block_h = min(height, max(1, int(round(height * h_scale))))
        return block_w, block_h

    def _frame_range(self, frames: int, device: torch.device) -> tuple[int, int]:
        if torch.rand((), device=device) < self.temporal_static_ratio:
            return 0, frames
        start_f = int(torch.randint(0, max(1, frames), (), device=device).item())
        end_f = int(torch.randint(start_f + 1, frames + 1, (), device=device).item())
        return start_f, end_f


RandomInpaintMaskGenerator = InpaintMaskGenerator


class WanInpaintTask(WanTrainingTask):
    name = "inpaint"

    def __init__(self, vae_mini_batch: int = 1, mask_generator: InpaintMaskGenerator | None = None) -> None:
        super().__init__(vae_mini_batch=vae_mini_batch)
        self.mask_generator = mask_generator or InpaintMaskGenerator()

    def build_condition(self, bundle, pixel_values: torch.Tensor, latents: torch.Tensor) -> TaskCondition:
        if bundle.clip_image_encoder is None:
            raise ValueError("task='inpaint' requires a CLIP image encoder in the Wan model bundle.")

        mask = self.mask_generator(pixel_values)
        masked_pixel_values = pixel_values * (1 - mask)
        mask_latents = encode_video_latents(bundle.vae, masked_pixel_values, mini_batch=self.vae_mini_batch)

        keep_mask = self._latent_keep_mask(mask, latents).to(dtype=latents.dtype)
        loss_mask = (1.0 - keep_mask).mean(dim=1, keepdim=True).clamp(0, 1).to(dtype=latents.dtype)
        y = torch.cat([keep_mask, mask_latents], dim=1)
        clip_fea = self._clip_features(bundle, pixel_values, latents.dtype)
        return TaskCondition(y=y, clip_fea=clip_fea, loss_mask=loss_mask)

    def _latent_keep_mask(self, mask: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        mask = mask.permute(0, 2, 1, 3, 4).contiguous()
        mask = torch.cat([torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), mask[:, :, 1:]], dim=2)
        mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
        return F.interpolate(1 - mask, size=list(latents.shape[2:]), mode="trilinear", align_corners=False)

    def _clip_features(self, bundle, pixel_values: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        clip_context = []
        for first_frame in pixel_values[:, 0]:
            clip_context.append(bundle.clip_image_encoder([first_frame[:, None, :, :]]))
        return torch.cat(clip_context, dim=0).to(dtype=dtype)


def build_training_task(cfg: dict, vae_mini_batch: int) -> WanTrainingTask:
    task_name = cfg.get("task", "t2v")
    if task_name == "t2v":
        return WanT2VTask(vae_mini_batch=vae_mini_batch)
    if task_name == "inpaint":
        return WanInpaintTask(
            vae_mini_batch=vae_mini_batch,
            mask_generator=InpaintMaskGenerator.from_config(cfg),
        )
    raise ValueError(f"Unsupported task: {task_name}")
