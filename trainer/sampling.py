from __future__ import annotations

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
from trainer.tasks import RandomInpaintMaskGenerator
from utils.video_io import save_video_preview


@dataclass
class TrainingSampleConfig:
    every_steps: int
    num_inference_steps: int
    guidance_scale: float
    seed: int
    sample_index: int
    sample_count: int
    fps: int
    save_history: bool
    negative_prompt: str

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "TrainingSampleConfig":
        return cls(
            every_steps=int(cfg.get("sample_every_steps", 0)),
            num_inference_steps=int(cfg.get("sample_num_inference_steps", 20)),
            guidance_scale=float(cfg.get("sample_guidance_scale", 6.0)),
            seed=int(cfg.get("sample_seed", 1234)),
            sample_index=int(cfg.get("sample_index", 0)),
            sample_count=max(1, int(cfg.get("sample_count", 1))),
            fps=int(cfg.get("sample_fps", 8)),
            save_history=bool(cfg.get("sample_save_history", True)),
            negative_prompt=str(cfg.get("sample_negative_prompt", "")),
        )


class TrainingVideoSampler:
    def __init__(
        self,
        cfg: dict[str, Any],
        output_dir: str,
        dataset: VideoCaptionDataset,
        sample_size: tuple[int, int],
        weight_dtype: torch.dtype,
    ) -> None:
        self.config = TrainingSampleConfig.from_dict(cfg)
        self.output_dir = Path(output_dir) / "samples"
        self.weight_dtype = weight_dtype
        self.sample_size = sample_size
        self.mask_generator = RandomInpaintMaskGenerator()
        self.samples = self._load_fixed_samples(dataset)

    @property
    def enabled(self) -> bool:
        return self.config.every_steps > 0

    def should_sample(self, global_step: int) -> bool:
        return self.enabled and global_step > 0 and global_step % self.config.every_steps == 0

    def _load_fixed_samples(self, dataset: VideoCaptionDataset) -> list[dict[str, Any]]:
        samples = []
        for sample_offset in range(self.config.sample_count):
            samples.append(self._load_fixed_sample(dataset, sample_offset))
        return samples

    def _load_fixed_sample(self, dataset: VideoCaptionDataset, sample_offset: int) -> dict[str, Any]:
        state = random.getstate()
        try:
            random.seed(self.config.seed + sample_offset)
            item = dataset[(self.config.sample_index + sample_offset) % len(dataset)]
        finally:
            random.setstate(state)
        batch = VideoCollator(self.sample_size, random_crop=False)([item])
        return batch

    @torch.no_grad()
    def sample_step(self, bundle, device: torch.device, global_step: int) -> list[Path]:
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

        latest_paths = []
        for sample_offset, sample in enumerate(self.samples):
            output = self._run_sample(pipeline, sample, device, sample_offset)
            latest_paths.append(self._save_sample_video(output, global_step, sample_offset))

        if was_training:
            transformer.train()
        return latest_paths

    def _run_sample(self, pipeline, sample: dict[str, Any], device: torch.device, sample_offset: int) -> torch.Tensor:
        pixel_values = sample["pixel_values"].to(device=device, dtype=self.weight_dtype)
        video = (pixel_values + 1.0) / 2.0
        video = video.permute(0, 2, 1, 3, 4).contiguous()
        mask_video = self._build_mask_video(pixel_values, device, self.config.seed + sample_offset)
        clip_image = self._first_frame_image(pixel_values[0, 0])
        generator = torch.Generator(device=device).manual_seed(self.config.seed + sample_offset)

        return pipeline(
            sample["text"][0],
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

    def _save_sample_video(self, output: torch.Tensor, global_step: int, sample_offset: int) -> Path:
        sample_name = f"sample_{sample_offset:02d}"
        latest_path = self.output_dir / f"latest_{sample_name}.mp4"
        if self.config.save_history:
            step_path = self.output_dir / f"step_{global_step:06d}_{sample_name}.mp4"
            save_video_preview(self._normalize_pipeline_video(output), step_path, fps=self.config.fps)
            shutil.copyfile(step_path, latest_path)
        else:
            save_video_preview(self._normalize_pipeline_video(output), latest_path, fps=self.config.fps)
        return latest_path

    def _build_mask_video(self, pixel_values: torch.Tensor, device: torch.device, seed: int) -> torch.Tensor:
        cpu_state = torch.random.get_rng_state()
        cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        try:
            torch.manual_seed(seed)
            if device.type == "cuda":
                torch.cuda.manual_seed(seed)
            mask = self.mask_generator(pixel_values)
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state(cuda_state, device)
        return (mask.permute(0, 2, 1, 3, 4).contiguous() * 255.0).to(device=device, dtype=self.weight_dtype)

    def _first_frame_image(self, frame: torch.Tensor) -> Image.Image:
        frame = ((frame.detach().float().cpu() + 1.0) / 2.0).clamp(0, 1)
        frame = (frame.permute(1, 2, 0).numpy() * 255).round().astype("uint8")
        return Image.fromarray(frame)

    def _normalize_pipeline_video(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"Expected generated video with 5 dims, got shape={tuple(video.shape)}")
        video = video[0]
        if video.shape[0] in (1, 3):
            return video.permute(1, 0, 2, 3).contiguous()
        if video.shape[-1] in (1, 3):
            return video.permute(0, 3, 1, 2).contiguous()
        raise ValueError(f"Could not infer generated video layout from shape={tuple(video.shape)}")
