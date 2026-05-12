from __future__ import annotations

import inspect
import os
from dataclasses import dataclass
from typing import Any

import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer

from third_party.videox_fun.models import AutoencoderKLWan, CLIPModel, WanT5EncoderModel, WanTransformer3DModel


def _filter_kwargs(cls: type, kwargs: dict[str, Any]) -> dict[str, Any]:
    params = inspect.signature(cls.__init__).parameters
    return {k: v for k, v in kwargs.items() if k in params}


@dataclass
class WanModelBundle:
    tokenizer: Any
    text_encoder: WanT5EncoderModel
    vae: AutoencoderKLWan
    transformer: WanTransformer3DModel
    scheduler: FlowMatchEulerDiscreteScheduler
    clip_image_encoder: CLIPModel | None = None

    def unwrap_transformer(self) -> WanTransformer3DModel:
        return self.transformer.module if hasattr(self.transformer, "module") else self.transformer

    def freeze_inference_modules(self) -> None:
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        if self.clip_image_encoder is not None:
            self.clip_image_encoder.requires_grad_(False)

    def to_inference_device(self, device: torch.device, dtype: torch.dtype) -> None:
        self.vae.to(device, dtype=dtype)
        self.text_encoder.to(device, dtype=dtype)
        if self.clip_image_encoder is not None:
            self.clip_image_encoder.to(device, dtype=dtype)
        self.vae.eval()
        self.text_encoder.eval()
        if self.clip_image_encoder is not None:
            self.clip_image_encoder.eval()

    def set_scheduler_device(self, device: torch.device) -> None:
        self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps, device=device)


WanT2VBundle = WanModelBundle


class WanModelLoader:
    def __init__(self, cfg: dict[str, Any], weight_dtype: torch.dtype) -> None:
        self.cfg = cfg
        self.weight_dtype = weight_dtype
        self.model_root = cfg["pretrained_model_path"]
        self.model_cfg = cfg["model"]

    def load(self) -> WanModelBundle:
        text_kwargs = self.model_cfg["text_encoder_kwargs"]
        vae_kwargs = self.model_cfg["vae_kwargs"]
        transformer_kwargs = self.model_cfg["transformer_additional_kwargs"]
        image_kwargs = self.model_cfg.get("image_encoder_kwargs", {})

        scheduler = self._load_scheduler()
        tokenizer = self._load_tokenizer(text_kwargs)
        text_encoder = self._load_text_encoder(text_kwargs)
        vae = self._load_vae(vae_kwargs)
        transformer = self._load_transformer(transformer_kwargs)
        clip_image_encoder = self._load_clip_image_encoder(image_kwargs)

        bundle = WanModelBundle(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            clip_image_encoder=clip_image_encoder,
        )
        bundle.freeze_inference_modules()
        return bundle

    def _load_scheduler(self) -> FlowMatchEulerDiscreteScheduler:
        scheduler_kwargs = self.model_cfg["scheduler_kwargs"].copy()
        return FlowMatchEulerDiscreteScheduler(
            **_filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
        )

    def _load_tokenizer(self, text_kwargs: dict[str, Any]) -> Any:
        tokenizer_subpath = text_kwargs.get("tokenizer_subpath", "tokenizer")
        local_path = os.path.join(self.model_root, tokenizer_subpath)
        return AutoTokenizer.from_pretrained(
            local_path if os.path.exists(local_path) else text_kwargs.get("tokenizer_subpath", "google/umt5-xxl")
        )

    def _load_text_encoder(self, text_kwargs: dict[str, Any]) -> WanT5EncoderModel:
        return WanT5EncoderModel.from_pretrained(
            os.path.join(self.model_root, text_kwargs.get("text_encoder_subpath", "text_encoder")),
            additional_kwargs=text_kwargs,
            low_cpu_mem_usage=True,
            torch_dtype=self.weight_dtype,
        ).eval()

    def _load_vae(self, vae_kwargs: dict[str, Any]) -> AutoencoderKLWan:
        return AutoencoderKLWan.from_pretrained(
            os.path.join(self.model_root, vae_kwargs.get("vae_subpath", "vae")),
            additional_kwargs=vae_kwargs,
        ).eval()

    def _load_transformer(self, transformer_kwargs: dict[str, Any]) -> WanTransformer3DModel:
        return WanTransformer3DModel.from_pretrained(
            os.path.join(self.model_root, transformer_kwargs.get("transformer_subpath", "transformer")),
            transformer_additional_kwargs=transformer_kwargs,
        ).to(self.weight_dtype)

    def _load_clip_image_encoder(self, image_kwargs: dict[str, Any]) -> CLIPModel | None:
        if self.cfg.get("task", "t2v") != "inpaint":
            return None
        return CLIPModel.from_pretrained(
            os.path.join(
                self.model_root,
                image_kwargs.get("image_encoder_subpath", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
            ),
        ).eval()


def load_wan_t2v_bundle(cfg: dict[str, Any], weight_dtype: torch.dtype) -> WanModelBundle:
    return WanModelLoader(cfg, weight_dtype).load()
