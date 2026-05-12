from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from tqdm.auto import tqdm

from data.datamodule import VideoDataModule
from models.wan_forward import encode_prompts, encode_video_latents, transformer_seq_len
from models.wan_loader import WanModelBundle, WanModelLoader
from trainer.checkpointing import resolve_resume_checkpoint
from trainer.metrics_logger import UHTKMetricsLogger
from trainer.sampling import TrainingVideoSampler
from trainer.schedulers import flow_matching_loss, get_sigmas, sample_flow_timesteps
from trainer.tasks import WanInpaintTask, WanT2VTask, WanTrainingTask, build_training_task
from utils.device import dtype_from_mixed_precision, seed_everything
from utils.paths import ensure_dir, expand_path
from utils.video_io import save_video_preview


class TrainableParameterSelector:
    def __init__(self, module_names: list[str]) -> None:
        self.module_names = module_names

    def select(self, transformer: torch.nn.Module) -> list[torch.nn.Parameter]:
        transformer.requires_grad_(False)
        train_all = "*" in self.module_names or "." in self.module_names
        for name, param in transformer.named_parameters():
            if train_all or any(module_name in name for module_name in self.module_names):
                param.requires_grad = True
        params = [p for p in transformer.parameters() if p.requires_grad]
        if not params:
            raise ValueError(f"No trainable parameters matched trainable_modules={self.module_names}")
        return params


class WanTrainer:
    def __init__(
        self,
        cfg: dict[str, Any],
        task: WanTrainingTask | None = None,
        data_module: VideoDataModule | None = None,
    ) -> None:
        self.cfg = cfg
        self.output_dir = expand_path(cfg["output_dir"], cfg.get("_config_dir"))
        self.accelerator = self._build_accelerator()
        self.weight_dtype = dtype_from_mixed_precision(self.accelerator.mixed_precision)
        self.task = task or build_training_task(cfg, vae_mini_batch=int(cfg.get("vae_mini_batch", 1)))
        self.data_module = data_module or VideoDataModule.from_config(cfg)
        self.bundle: WanModelBundle | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.trainable_params: list[torch.nn.Parameter] = []
        self.metrics_logger: UHTKMetricsLogger | None = None
        self.video_sampler: TrainingVideoSampler | None = None

    def train(self) -> None:
        self._setup()
        assert self.bundle is not None
        assert self.optimizer is not None

        resume_path = resolve_resume_checkpoint(self.output_dir, self.cfg.get("resume_from_checkpoint"))
        global_step = self._resume_global_step(resume_path)
        progress = tqdm(
            range(global_step, int(self.cfg["max_train_steps"])),
            disable=not self.accelerator.is_local_main_process,
            desc="Steps",
        )

        max_train_steps = int(self.cfg["max_train_steps"])
        checkpointing_steps = int(self.cfg.get("checkpointing_steps", 1000))
        step_sleep_seconds = float(self.cfg.get("step_sleep_seconds", 0.0))
        text_length = int(self.cfg["model"]["text_encoder_kwargs"].get("text_length", 512))
        step_start_time = time.perf_counter()

        while global_step < max_train_steps:
            for batch in self.train_dataloader:
                if global_step >= max_train_steps:
                    break

                loss, metrics = self._train_step(batch, global_step=global_step, text_length=text_length)

                if self.accelerator.sync_gradients:
                    if self.accelerator.device.type == "cuda":
                        torch.cuda.synchronize(self.accelerator.device)
                    step_time_sec = time.perf_counter() - step_start_time
                    global_step += 1
                    progress.update(1)
                    self._log_metrics(loss, global_step, metrics, step_time_sec)
                    progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")
                    self._save_checkpoint_if_due(global_step, checkpointing_steps)
                    self._sample_video_if_due(global_step)
                    self._sleep_between_steps(step_sleep_seconds)
                    step_start_time = time.perf_counter()

        self._finish()

    def _build_accelerator(self) -> Accelerator:
        return Accelerator(
            gradient_accumulation_steps=int(self.cfg.get("gradient_accumulation_steps", 1)),
            mixed_precision=self.cfg.get("mixed_precision", "no"),
            project_dir=self.output_dir,
        )

    def _setup(self) -> None:
        ensure_dir(self.output_dir)
        if self.accelerator.is_main_process:
            ensure_dir(os.path.join(self.output_dir, "sanity"))
        seed_everything(int(self.cfg.get("seed", 42)) + self.accelerator.process_index)

        self.bundle = WanModelLoader(self.cfg, self.weight_dtype).load()
        if self.cfg.get("gradient_checkpointing", False):
            self._enable_gradient_checkpointing(self.bundle.transformer)

        self.trainable_params = TrainableParameterSelector(
            self.cfg.get("trainable_modules", ["*"])
        ).select(self.bundle.transformer)
        self.optimizer = self._build_optimizer(self.trainable_params)
        self.train_dataloader = self.data_module.train_dataloader(self.accelerator)
        if self.accelerator.is_main_process:
            self.video_sampler = TrainingVideoSampler(
                self.cfg,
                output_dir=self.output_dir,
                dataset=self.data_module.train_dataset,
                sample_size=self.data_module.config.sample_size,
                weight_dtype=self.weight_dtype,
            )

        self.bundle.transformer, self.optimizer, self.train_dataloader = self.accelerator.prepare(
            self.bundle.transformer,
            self.optimizer,
            self.train_dataloader,
        )
        self.bundle.to_inference_device(self.accelerator.device, self.weight_dtype)
        self.bundle.set_scheduler_device(self.accelerator.device)

        if self.accelerator.is_main_process:
            self.metrics_logger = UHTKMetricsLogger(
                self.cfg,
                self.output_dir,
                enabled=bool(self.cfg.get("visualizer_enabled", True)),
            )

    def _build_optimizer(self, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params,
            lr=float(self.cfg["learning_rate"]),
            betas=(0.9, 0.999),
            weight_decay=float(self.cfg.get("adam_weight_decay", 0.0)),
            eps=float(self.cfg.get("adam_epsilon", 1e-8)),
        )

    def _enable_gradient_checkpointing(self, transformer: torch.nn.Module) -> None:
        try:
            transformer.enable_gradient_checkpointing()
        except TypeError as exc:
            if "unexpected keyword argument 'enable'" not in str(exc):
                raise
            transformer._set_gradient_checkpointing(transformer, value=True)

    def _resume_global_step(self, resume_path: str | None) -> int:
        if not resume_path:
            return 0
        self.accelerator.print(f"Resuming from checkpoint: {resume_path}")
        self.accelerator.load_state(resume_path)
        try:
            return int(Path(resume_path).name.split("-")[-1])
        except Exception:
            return 0

    def _train_step(self, batch: dict[str, Any], global_step: int, text_length: int) -> tuple[torch.Tensor, dict[str, float]]:
        assert self.bundle is not None
        assert self.optimizer is not None

        with self.accelerator.accumulate(self.bundle.transformer):
            pixel_values = batch["pixel_values"].to(self.accelerator.device, dtype=self.weight_dtype)
            if global_step == 0 and self.accelerator.is_main_process:
                save_video_preview(pixel_values[0], os.path.join(self.output_dir, "sanity", "train_batch_0.mp4"))

            latents, prompt_embeds = self._encode_inputs(pixel_values, batch["text"], text_length)
            noise = torch.randn_like(latents)
            _, timesteps = sample_flow_timesteps(
                self.bundle.scheduler,
                batch_size=latents.shape[0],
                device=self.accelerator.device,
            )
            sigmas = get_sigmas(
                self.bundle.scheduler,
                timesteps,
                n_dim=latents.ndim,
                dtype=latents.dtype,
                device=self.accelerator.device,
            )
            noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
            target = noise - latents
            condition = self._build_task_condition(pixel_values, latents)
            seq_len = transformer_seq_len(self.bundle.unwrap_transformer(), self.bundle.vae, latents)

            with torch.autocast(device_type="cuda", dtype=self.weight_dtype, enabled=self.accelerator.device.type == "cuda"):
                noise_pred = self.bundle.transformer(
                    x=noisy_latents,
                    context=prompt_embeds,
                    t=timesteps,
                    seq_len=seq_len,
                    y=condition.y,
                    clip_fea=condition.clip_fea,
                )
                loss = flow_matching_loss(noise_pred, target)

            self.accelerator.backward(loss)
            grad_norm = None
            if self.accelerator.sync_gradients:
                grad_norm = self.accelerator.clip_grad_norm_(
                    self.trainable_params,
                    float(self.cfg.get("max_grad_norm", 1.0)),
                )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            metrics = {
                "timestep_mean": timesteps.detach().float().mean().item(),
                "sigma_mean": sigmas.detach().float().mean().item(),
            }
            if grad_norm is not None:
                metrics["grad_norm"] = grad_norm.detach().float().item() if hasattr(grad_norm, "detach") else float(grad_norm)
            return loss, metrics

    def _encode_inputs(
        self,
        pixel_values: torch.Tensor,
        texts: list[str],
        text_length: int,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        assert self.bundle is not None
        with torch.no_grad():
            latents = encode_video_latents(
                self.bundle.vae,
                pixel_values,
                mini_batch=int(self.cfg.get("vae_mini_batch", 1)),
            )
            prompt_embeds = encode_prompts(
                self.bundle.tokenizer,
                self.bundle.text_encoder,
                texts,
                device=self.accelerator.device,
                max_length=text_length,
            )
        return latents, prompt_embeds

    def _build_task_condition(self, pixel_values: torch.Tensor, latents: torch.Tensor):
        assert self.bundle is not None
        with torch.no_grad():
            return self.task.build_condition(self.bundle, pixel_values, latents)

    def _save_checkpoint_if_due(self, global_step: int, checkpointing_steps: int) -> None:
        if checkpointing_steps <= 0 or global_step % checkpointing_steps != 0:
            return
        assert self.bundle is not None
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{global_step}")
        self.accelerator.save_state(checkpoint_dir)
        if self.accelerator.is_main_process:
            self.bundle.unwrap_transformer().save_pretrained(os.path.join(checkpoint_dir, "transformer"))

    def _sample_video_if_due(self, global_step: int) -> None:
        if not self.accelerator.is_main_process or self.video_sampler is None:
            return
        if not self.video_sampler.should_sample(global_step):
            return
        assert self.bundle is not None
        latest_path = self.video_sampler.sample_step(self.bundle, self.accelerator.device, global_step)
        self.accelerator.print(f"Saved training sample: {latest_path}")

    def _sleep_between_steps(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        time.sleep(seconds)

    def _log_metrics(
        self,
        loss: torch.Tensor,
        global_step: int,
        step_metrics: dict[str, float],
        step_time_sec: float,
    ) -> None:
        if not self.accelerator.is_main_process or self.metrics_logger is None:
            return
        assert self.optimizer is not None
        effective_batch_size = (
            int(self.cfg.get("train_batch_size", 1))
            * int(self.cfg.get("gradient_accumulation_steps", 1))
            * self.accelerator.num_processes
        )
        metrics = {
            "train_loss": loss.detach().float().item(),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
            "step_time_sec": step_time_sec,
            "samples_per_sec": effective_batch_size / max(step_time_sec, 1e-12),
            "samples_seen": global_step * effective_batch_size,
            **step_metrics,
        }
        if self.accelerator.device.type == "cuda":
            device = self.accelerator.device
            metrics.update(
                {
                    "gpu_memory_allocated_gb": torch.cuda.memory_allocated(device) / 1024**3,
                    "gpu_memory_reserved_gb": torch.cuda.memory_reserved(device) / 1024**3,
                    "gpu_memory_peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 1024**3,
                }
            )
            torch.cuda.reset_peak_memory_stats(device)
        self.metrics_logger.log(metrics, step=global_step)

    def _finish(self) -> None:
        assert self.bundle is not None
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            final_dir = os.path.join(self.output_dir, "final", "transformer")
            self.bundle.unwrap_transformer().save_pretrained(final_dir)
            if self.metrics_logger is not None:
                self.metrics_logger.close()
            self.accelerator.end_training()


def train_wan(cfg: dict[str, Any]) -> None:
    build_wan_trainer(cfg).train()


class WanT2VTrainer(WanTrainer):
    def __init__(self, cfg: dict[str, Any], data_module: VideoDataModule | None = None) -> None:
        super().__init__(
            cfg,
            task=WanT2VTask(vae_mini_batch=int(cfg.get("vae_mini_batch", 1))),
            data_module=data_module,
        )


class WanInpaintTrainer(WanTrainer):
    def __init__(self, cfg: dict[str, Any], data_module: VideoDataModule | None = None) -> None:
        super().__init__(
            cfg,
            task=WanInpaintTask(vae_mini_batch=int(cfg.get("vae_mini_batch", 1))),
            data_module=data_module,
        )


def build_wan_trainer(cfg: dict[str, Any]) -> WanTrainer:
    task_name = cfg.get("task", "t2v")
    if task_name == "t2v":
        return WanT2VTrainer(cfg)
    if task_name == "inpaint":
        return WanInpaintTrainer(cfg)
    return WanTrainer(cfg, task=build_training_task(cfg, vae_mini_batch=int(cfg.get("vae_mini_batch", 1))))
