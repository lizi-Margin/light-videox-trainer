from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.wan_forward import encode_video_latents


@dataclass
class TaskCondition:
    y: torch.Tensor | None
    clip_fea: torch.Tensor | None


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


class RandomInpaintMaskGenerator:
    def __call__(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch, frames, _, height, width = pixel_values.shape
        mask = pixel_values.new_zeros(batch, frames, 1, height, width)
        for idx in range(batch):
            if torch.rand((), device=pixel_values.device) < 0.2:
                mask[idx] = 1
                continue
            block_w = int(torch.randint(max(1, width // 4), max(2, width * 3 // 4), (), device=pixel_values.device).item())
            block_h = int(torch.randint(max(1, height // 4), max(2, height * 3 // 4), (), device=pixel_values.device).item())
            left = int(torch.randint(0, max(1, width - block_w + 1), (), device=pixel_values.device).item())
            top = int(torch.randint(0, max(1, height - block_h + 1), (), device=pixel_values.device).item())
            if torch.rand((), device=pixel_values.device) < 0.5:
                start_f, end_f = 0, frames
            else:
                start_f = int(torch.randint(0, max(1, frames), (), device=pixel_values.device).item())
                end_f = int(torch.randint(start_f + 1, frames + 1, (), device=pixel_values.device).item())
            mask[idx, start_f:end_f, :, top : top + block_h, left : left + block_w] = 1
        return mask


class WanInpaintTask(WanTrainingTask):
    name = "inpaint"

    def __init__(self, vae_mini_batch: int = 1, mask_generator: RandomInpaintMaskGenerator | None = None) -> None:
        super().__init__(vae_mini_batch=vae_mini_batch)
        self.mask_generator = mask_generator or RandomInpaintMaskGenerator()

    def build_condition(self, bundle, pixel_values: torch.Tensor, latents: torch.Tensor) -> TaskCondition:
        if bundle.clip_image_encoder is None:
            raise ValueError("task='inpaint' requires a CLIP image encoder in the Wan model bundle.")

        mask = self.mask_generator(pixel_values)
        masked_pixel_values = pixel_values * (1 - mask)
        mask_latents = encode_video_latents(bundle.vae, masked_pixel_values, mini_batch=self.vae_mini_batch)

        mask = self._latent_mask(mask, latents).to(dtype=latents.dtype)
        y = torch.cat([mask, mask_latents], dim=1)
        clip_fea = self._clip_features(bundle, pixel_values, latents.dtype)
        return TaskCondition(y=y, clip_fea=clip_fea)

    def _latent_mask(self, mask: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
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
        return WanInpaintTask(vae_mini_batch=vae_mini_batch)
    raise ValueError(f"Unsupported task: {task_name}")
