from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.collate import VideoCollator
from data.video_dataset import VideoCaptionDataset
from models.wan_forward import encode_prompts, encode_video_latents, transformer_seq_len
from models.wan_loader import load_wan_t2v_bundle
from trainer.checkpointing import resolve_resume_checkpoint
from trainer.schedulers import flow_matching_loss, get_sigmas, sample_flow_timesteps
from utils.device import dtype_from_mixed_precision, seed_everything
from utils.paths import ensure_dir, expand_path
from utils.video_io import save_video_preview

logger = get_logger(__name__)


def _sample_size(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid video_sample_size: {value}")


def _set_trainable(transformer: torch.nn.Module, trainable_modules: list[str]) -> list[torch.nn.Parameter]:
    transformer.requires_grad_(False)
    train_all = "*" in trainable_modules or "." in trainable_modules
    for name, param in transformer.named_parameters():
        if train_all or any(module_name in name for module_name in trainable_modules):
            param.requires_grad = True
    params = [p for p in transformer.parameters() if p.requires_grad]
    if not params:
        raise ValueError(f"No trainable parameters matched trainable_modules={trainable_modules}")
    return params


def _enable_gradient_checkpointing(transformer: torch.nn.Module) -> None:
    try:
        transformer.enable_gradient_checkpointing()
    except TypeError as exc:
        if "unexpected keyword argument 'enable'" not in str(exc):
            raise
        # The vendored VideoX-Fun Wan model uses the older Diffusers hook
        # signature: _set_gradient_checkpointing(module, value=False).
        transformer._set_gradient_checkpointing(transformer, value=True)


def _tracker_config(cfg: dict[str, Any]) -> dict[str, Any]:
    allowed = (int, float, str, bool, torch.Tensor)
    result: dict[str, Any] = {}
    for key, value in cfg.items():
        if value is None:
            result[key] = "null"
        elif isinstance(value, allowed):
            result[key] = value
        else:
            result[key] = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return result


def _zero_i2v_condition(transformer: torch.nn.Module, latents: torch.Tensor) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    transformer = transformer.module if hasattr(transformer, "module") else transformer
    if getattr(transformer, "model_type", "t2v") != "i2v":
        return None, None

    in_channels = int(transformer.config.in_channels)
    condition_channels = in_channels - int(latents.shape[1])
    if condition_channels <= 0:
        return None, None

    y = latents.new_zeros(
        latents.shape[0],
        condition_channels,
        latents.shape[2],
        latents.shape[3],
        latents.shape[4],
    )
    clip_fea = latents.new_zeros(latents.shape[0], 257, 1280)
    return y, clip_fea


def _resize_mask(mask: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
    target_size = list(latents.shape[2:])
    return F.interpolate(mask, size=target_size, mode="trilinear", align_corners=False)


def _random_video_mask(pixel_values: torch.Tensor) -> torch.Tensor:
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


def _inpaint_condition(bundle, pixel_values: torch.Tensor, latents: torch.Tensor, vae_mini_batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    if bundle.clip_image_encoder is None:
        raise ValueError("task='inpaint' requires clip_image_encoder in the loaded model bundle.")

    mask = _random_video_mask(pixel_values)
    masked_pixel_values = pixel_values * (1 - mask)
    mask_latents = encode_video_latents(bundle.vae, masked_pixel_values, mini_batch=vae_mini_batch)

    mask = mask.permute(0, 2, 1, 3, 4).contiguous()
    mask = torch.cat([torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), mask[:, :, 1:]], dim=2)
    mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]).transpose(1, 2)
    mask = _resize_mask(1 - mask, latents).to(dtype=latents.dtype)
    y = torch.cat([mask, mask_latents], dim=1)

    clip_context = []
    for first_frame in pixel_values[:, 0]:
        clip_context.append(bundle.clip_image_encoder([first_frame[:, None, :, :]]))
    clip_fea = torch.cat(clip_context, dim=0).to(dtype=latents.dtype)
    return y, clip_fea


def _build_dataloader(cfg: dict, accelerator: Accelerator) -> DataLoader:
    config_dir = cfg.get("_config_dir")
    metadata = expand_path(cfg["train_metadata"], config_dir)
    data_root = expand_path(cfg.get("train_data_root") or None, config_dir)
    sample_size = _sample_size(cfg["video_sample_size"])
    dataset = VideoCaptionDataset(
        metadata_path=metadata,
        data_root=data_root,
        sample_n_frames=int(cfg["video_sample_n_frames"]),
        sample_stride=int(cfg["video_sample_stride"]),
        sample_size=sample_size,
        text_drop_ratio=float(cfg.get("text_drop_ratio", 0.0)),
    )
    if accelerator.is_main_process:
        logger.info("Dataset size: %s", len(dataset))
    collator = VideoCollator(sample_size=sample_size, random_crop=bool(cfg.get("random_crop", True)))
    return DataLoader(
        dataset,
        batch_size=int(cfg["train_batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("dataloader_num_workers", 0)),
        persistent_workers=bool(cfg.get("dataloader_num_workers", 0)),
        collate_fn=collator,
        drop_last=True,
    )


def train(cfg: dict) -> None:
    output_dir = expand_path(cfg["output_dir"], cfg.get("_config_dir"))
    ensure_dir(output_dir)

    accelerator = Accelerator(
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        mixed_precision=cfg.get("mixed_precision", "no"),
        log_with="tensorboard",
        project_dir=output_dir,
    )
    if accelerator.is_main_process:
        ensure_dir(os.path.join(output_dir, "sanity"))
    seed_everything(int(cfg.get("seed", 42)) + accelerator.process_index)

    weight_dtype = dtype_from_mixed_precision(accelerator.mixed_precision)
    bundle = load_wan_t2v_bundle(cfg, weight_dtype=weight_dtype)

    if cfg.get("gradient_checkpointing", False):
        _enable_gradient_checkpointing(bundle.transformer)

    trainable_params = _set_trainable(bundle.transformer, cfg.get("trainable_modules", ["*"]))
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(cfg.get("adam_weight_decay", 0.0)),
        eps=float(cfg.get("adam_epsilon", 1e-8)),
    )

    train_dataloader = _build_dataloader(cfg, accelerator)
    bundle.transformer, optimizer, train_dataloader = accelerator.prepare(
        bundle.transformer, optimizer, train_dataloader
    )

    bundle.vae.to(accelerator.device, dtype=weight_dtype)
    bundle.text_encoder.to(accelerator.device, dtype=weight_dtype)
    if bundle.clip_image_encoder is not None:
        bundle.clip_image_encoder.to(accelerator.device, dtype=weight_dtype)
    bundle.vae.eval()
    bundle.text_encoder.eval()
    if bundle.clip_image_encoder is not None:
        bundle.clip_image_encoder.eval()
    bundle.scheduler.set_timesteps(bundle.scheduler.config.num_train_timesteps, device=accelerator.device)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            cfg.get("tracker_project_name", "light-videox-trainer"),
            config=_tracker_config(cfg),
        )

    resume_path = resolve_resume_checkpoint(output_dir, cfg.get("resume_from_checkpoint"))
    global_step = 0
    if resume_path:
        accelerator.print(f"Resuming from checkpoint: {resume_path}")
        accelerator.load_state(resume_path)
        try:
            global_step = int(Path(resume_path).name.split("-")[-1])
        except Exception:
            global_step = 0

    progress = tqdm(
        range(global_step, int(cfg["max_train_steps"])),
        disable=not accelerator.is_local_main_process,
        desc="Steps",
    )
    max_train_steps = int(cfg["max_train_steps"])
    checkpointing_steps = int(cfg.get("checkpointing_steps", 1000))
    vae_mini_batch = int(cfg.get("vae_mini_batch", 1))
    text_length = int(cfg["model"]["text_encoder_kwargs"].get("text_length", 512))

    while global_step < max_train_steps:
        for batch in train_dataloader:
            if global_step >= max_train_steps:
                break

            with accelerator.accumulate(bundle.transformer):
                pixel_values = batch["pixel_values"].to(accelerator.device, dtype=weight_dtype)

                if global_step == 0 and accelerator.is_main_process:
                    save_video_preview(pixel_values[0], os.path.join(output_dir, "sanity", "train_batch_0.mp4"))

                with torch.no_grad():
                    latents = encode_video_latents(bundle.vae, pixel_values, mini_batch=vae_mini_batch)
                    prompt_embeds = encode_prompts(
                        bundle.tokenizer,
                        bundle.text_encoder,
                        batch["text"],
                        device=accelerator.device,
                        max_length=text_length,
                    )

                noise = torch.randn_like(latents)
                _, timesteps = sample_flow_timesteps(
                    bundle.scheduler,
                    batch_size=latents.shape[0],
                    device=accelerator.device,
                )
                sigmas = get_sigmas(
                    bundle.scheduler,
                    timesteps,
                    n_dim=latents.ndim,
                    dtype=latents.dtype,
                    device=accelerator.device,
                )
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                target = noise - latents
                seq_len = transformer_seq_len(
                    accelerator.unwrap_model(bundle.transformer),
                    bundle.vae,
                    latents,
                )
                if cfg.get("task", "t2v") == "inpaint":
                    with torch.no_grad():
                        condition_y, clip_fea = _inpaint_condition(bundle, pixel_values, latents, vae_mini_batch)
                else:
                    condition_y, clip_fea = _zero_i2v_condition(
                        accelerator.unwrap_model(bundle.transformer),
                        latents,
                    )

                with torch.autocast(device_type="cuda", dtype=weight_dtype, enabled=accelerator.device.type == "cuda"):
                    noise_pred = bundle.transformer(
                        x=noisy_latents,
                        context=prompt_embeds,
                        t=timesteps,
                        seq_len=seq_len,
                        y=condition_y,
                        clip_fea=clip_fea,
                    )
                    loss = flow_matching_loss(noise_pred, target)

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, float(cfg.get("max_grad_norm", 1.0)))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress.update(1)
                accelerator.log({"train_loss": loss.detach().float().item()}, step=global_step)
                progress.set_postfix(loss=f"{loss.detach().float().item():.4f}")

                if checkpointing_steps > 0 and global_step % checkpointing_steps == 0:
                    checkpoint_dir = os.path.join(output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(checkpoint_dir)
                    if accelerator.is_main_process:
                        unwrapped = accelerator.unwrap_model(bundle.transformer)
                        unwrapped.save_pretrained(os.path.join(checkpoint_dir, "transformer"))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = os.path.join(output_dir, "final", "transformer")
        accelerator.unwrap_model(bundle.transformer).save_pretrained(final_dir)
        accelerator.end_training()
