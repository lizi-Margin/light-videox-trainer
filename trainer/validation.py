from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image

from data.collate import VideoCollator
from data.video_dataset import VideoCaptionDataset
from third_party.videox_fun.pipeline.pipeline_wan_fun_inpaint import WanFunInpaintPipeline
from trainer.tasks import InpaintMaskGenerator
from utils.paths import expand_path
from utils.video_io import save_video_preview


@dataclass
class ValidationConfig:
    every_steps: int
    metadata_path: str | None
    data_root: str | None
    count: int
    index: int
    seed: int
    num_inference_steps: int
    guidance_scale: float
    fps: int
    save_history: bool
    negative_prompt: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "ValidationConfig":
        config_dir = cfg.get("_config_dir")
        return cls(
            every_steps=int(cfg.get("validation_every_steps", 0)),
            metadata_path=expand_path(cfg.get("validation_metadata") or None, config_dir),
            data_root=expand_path(cfg.get("validation_data_root") or cfg.get("train_data_root") or None, config_dir),
            count=max(1, int(cfg.get("validation_count", 8))),
            index=int(cfg.get("validation_index", 0)),
            seed=int(cfg.get("validation_seed", 2026)),
            num_inference_steps=int(cfg.get("validation_num_inference_steps", cfg.get("sample_num_inference_steps", 20))),
            guidance_scale=float(cfg.get("validation_guidance_scale", cfg.get("sample_guidance_scale", 6.0))),
            fps=int(cfg.get("validation_fps", cfg.get("sample_fps", 8))),
            save_history=bool(cfg.get("validation_save_history", True)),
            negative_prompt=str(cfg.get("validation_negative_prompt", cfg.get("sample_negative_prompt", ""))),
        )


class TrainingVideoValidator:
    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: str,
        train_dataset: VideoCaptionDataset,
        sample_size: tuple[int, int],
        sample_n_frames: int,
        sample_stride: int,
        weight_dtype: torch.dtype,
    ) -> None:
        self.config = ValidationConfig.from_dict(cfg)
        self.output_dir = Path(output_dir) / "validation"
        self.sample_size = sample_size
        self.weight_dtype = weight_dtype
        self.mask_generator = InpaintMaskGenerator.from_config(cfg, prefix="validation_mask")
        if not self.enabled:
            self.cases = []
            return
        dataset = self._build_dataset(train_dataset, sample_n_frames, sample_stride)
        self.cases = self._load_cases(dataset)

    @property
    def enabled(self) -> bool:
        return self.config.every_steps > 0

    def should_validate(self, global_step: int) -> bool:
        return self.enabled and global_step > 0 and global_step % self.config.every_steps == 0

    def _build_dataset(
        self,
        train_dataset: VideoCaptionDataset,
        sample_n_frames: int,
        sample_stride: int,
    ) -> VideoCaptionDataset:
        if not self.config.metadata_path:
            return train_dataset
        return VideoCaptionDataset(
            metadata_path=self.config.metadata_path,
            data_root=self.config.data_root,
            sample_n_frames=sample_n_frames,
            sample_stride=sample_stride,
            sample_size=self.sample_size,
            text_drop_ratio=0.0,
        )

    def _load_cases(self, dataset: VideoCaptionDataset) -> list[dict[str, Any]]:
        cases = []
        for case_idx in range(self.config.count):
            state = random.getstate()
            try:
                random.seed(self.config.seed + case_idx)
                item = dataset[(self.config.index + case_idx) % len(dataset)]
            finally:
                random.setstate(state)
            batch = VideoCollator(self.sample_size, random_crop=False)([item])
            cases.append(batch)
        return cases

    @torch.no_grad()
    def validate_step(self, bundle, device: torch.device, global_step: int) -> dict[str, float]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        transformer = bundle.unwrap_transformer()
        was_training = transformer.training
        transformer.eval()

        pipeline = WanFunInpaintPipeline(
            tokenizer=bundle.tokenizer,
            text_encoder=bundle.text_encoder,
            vae=bundle.vae,
            transformer=transformer,
            clip_image_encoder=bundle.clip_image_encoder,
            scheduler=FlowMatchEulerDiscreteScheduler.from_config(bundle.scheduler.config),
        ).to(device)

        step_dir = self._step_dir(global_step)
        latest_dir = self.output_dir / "latest"
        if latest_dir.exists():
            shutil.rmtree(latest_dir)
        latest_dir.mkdir(parents=True, exist_ok=True)

        metrics: list[dict[str, float]] = []
        for case_idx, case in enumerate(self.cases):
            case_metrics = self._run_case(pipeline, case, device, step_dir, latest_dir, case_idx)
            metrics.append(case_metrics)

        summary = self._summarize(metrics)
        summary_payload = {"mean": summary, "cases": metrics}
        summary_path = step_dir / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        shutil.copyfile(summary_path, latest_dir / "summary.json")

        if was_training:
            transformer.train()
        return summary

    def _step_dir(self, global_step: int) -> Path:
        if self.config.save_history:
            return self.output_dir / f"step_{global_step:06d}"
        return self.output_dir / "latest_step"

    def _run_case(
        self,
        pipeline,
        case: dict[str, Any],
        device: torch.device,
        step_dir: Path,
        latest_dir: Path,
        case_idx: int,
    ) -> dict[str, float]:
        pixel_values = case["pixel_values"].to(device=device, dtype=self.weight_dtype)
        mask = self._build_mask(pixel_values, device, self.config.seed + case_idx)
        masked = pixel_values * (1.0 - mask)
        output = self._run_pipeline(pipeline, case, pixel_values, mask, device, case_idx)
        generated = self._normalize_pipeline_video(output).to(device=device, dtype=self.weight_dtype)

        case_dir = step_dir / f"case_{case_idx:03d}"
        latest_case_dir = latest_dir / f"case_{case_idx:03d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        latest_case_dir.mkdir(parents=True, exist_ok=True)

        self._save_case_assets(case_dir, pixel_values[0], mask[0], masked[0], generated, case, case_idx)
        if latest_case_dir.exists():
            shutil.rmtree(latest_case_dir)
        shutil.copytree(case_dir, latest_case_dir)

        return self._case_metrics(pixel_values[0], generated, mask[0])

    def _run_pipeline(
        self,
        pipeline,
        case: dict[str, Any],
        pixel_values: torch.Tensor,
        mask: torch.Tensor,
        device: torch.device,
        case_idx: int,
    ) -> torch.Tensor:
        video = (pixel_values + 1.0) / 2.0
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        mask_video = (mask.permute(0, 2, 1, 3, 4).contiguous() * 255.0).to(device=device, dtype=self.weight_dtype)
        clip_image = self._first_frame_image(pixel_values[0, 0])
        generator = torch.Generator(device=device).manual_seed(self.config.seed + case_idx)
        return pipeline(
            case["text"][0],
            negative_prompt=self.config.negative_prompt,
            height=self.sample_size[0],
            width=self.sample_size[1],
            video=video,
            mask_video=mask_video,
            num_frames=pixel_values.shape[1],
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            generator=generator,
            clip_image=clip_image,
            output_type="numpy",
            return_dict=False,
            zero_out_mask_region=True,
        ).videos

    def _build_mask(self, pixel_values: torch.Tensor, device: torch.device, seed: int) -> torch.Tensor:
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        try:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            return self.mask_generator(pixel_values)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)

    def _save_case_assets(
        self,
        case_dir: Path,
        source: torch.Tensor,
        mask: torch.Tensor,
        masked: torch.Tensor,
        generated: torch.Tensor,
        case: dict[str, Any],
        case_idx: int,
    ) -> None:
        save_video_preview(source, case_dir / "source.mp4", fps=self.config.fps)
        save_video_preview(mask.expand(-1, 3, -1, -1), case_dir / "mask.mp4", fps=self.config.fps)
        save_video_preview(masked, case_dir / "masked_input.mp4", fps=self.config.fps)
        save_video_preview(generated, case_dir / "generated.mp4", fps=self.config.fps)
        meta = {
            "text": case["text"][0],
            "file_path": case["file_path"][0],
            "case_index": case_idx,
            "seed": self.config.seed + case_idx,
            "num_inference_steps": self.config.num_inference_steps,
            "guidance_scale": self.config.guidance_scale,
        }
        (case_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def _case_metrics(self, source: torch.Tensor, generated: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
        generated = generated.to(device=source.device, dtype=source.dtype)
        mask_3c = mask.expand_as(source).to(dtype=source.dtype)
        source01 = (source.float() + 1.0) / 2.0
        generated01 = generated.float().clamp(0, 1)
        diff2 = (generated01 - source01).pow(2)
        masked_mse = (diff2 * mask_3c).sum() / mask_3c.sum().clamp_min(1.0)
        full_mse = diff2.mean()
        return {
            "validation_masked_mse": masked_mse.item(),
            "validation_masked_psnr": self._psnr(masked_mse).item(),
            "validation_full_mse": full_mse.item(),
            "validation_full_psnr": self._psnr(full_mse).item(),
            "validation_mask_fraction": mask.float().mean().item(),
        }

    def _summarize(self, metrics: list[dict[str, float]]) -> dict[str, float]:
        keys = metrics[0].keys()
        return {key: sum(item[key] for item in metrics) / len(metrics) for key in keys}

    def _psnr(self, mse: torch.Tensor) -> torch.Tensor:
        return -10.0 * torch.log10(mse.clamp_min(1e-12))

    def _first_frame_image(self, frame: torch.Tensor) -> Image.Image:
        frame = ((frame.detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
        frame = (frame.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        return Image.fromarray(frame)

    def _normalize_pipeline_video(self, video: torch.Tensor) -> torch.Tensor:
        if not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)
        if video.ndim != 5:
            raise ValueError(f"Expected generated video with 5 dims, got shape={tuple(video.shape)}")
        video = video[0]
        if video.shape[0] in (1, 3):
            return video.permute(1, 0, 2, 3).contiguous()
        if video.shape[-1] in (1, 3):
            return video.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Could not infer generated video layout from shape={tuple(video.shape)}")
