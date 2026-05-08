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
class WanT2VBundle:
    tokenizer: Any
    text_encoder: WanT5EncoderModel
    vae: AutoencoderKLWan
    transformer: WanTransformer3DModel
    scheduler: FlowMatchEulerDiscreteScheduler
    clip_image_encoder: CLIPModel | None = None


def load_wan_t2v_bundle(cfg: dict[str, Any], weight_dtype: torch.dtype) -> WanT2VBundle:
    model_root = cfg["pretrained_model_path"]
    model_cfg = cfg["model"]
    text_kwargs = model_cfg["text_encoder_kwargs"]
    vae_kwargs = model_cfg["vae_kwargs"]
    transformer_kwargs = model_cfg["transformer_additional_kwargs"]
    image_kwargs = model_cfg.get("image_encoder_kwargs", {})

    scheduler_kwargs = model_cfg["scheduler_kwargs"].copy()
    scheduler = FlowMatchEulerDiscreteScheduler(
        **_filter_kwargs(FlowMatchEulerDiscreteScheduler, scheduler_kwargs)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(model_root, text_kwargs.get("tokenizer_subpath", "tokenizer"))
        if os.path.exists(os.path.join(model_root, text_kwargs.get("tokenizer_subpath", "tokenizer")))
        else text_kwargs.get("tokenizer_subpath", "google/umt5-xxl")
    )

    text_encoder = WanT5EncoderModel.from_pretrained(
        os.path.join(model_root, text_kwargs.get("text_encoder_subpath", "text_encoder")),
        additional_kwargs=text_kwargs,
        low_cpu_mem_usage=True,
        torch_dtype=weight_dtype,
    ).eval()

    vae = AutoencoderKLWan.from_pretrained(
        os.path.join(model_root, vae_kwargs.get("vae_subpath", "vae")),
        additional_kwargs=vae_kwargs,
    ).eval()

    transformer = WanTransformer3DModel.from_pretrained(
        os.path.join(model_root, transformer_kwargs.get("transformer_subpath", "transformer")),
        transformer_additional_kwargs=transformer_kwargs,
    ).to(weight_dtype)

    clip_image_encoder = None
    if cfg.get("task", "t2v") == "inpaint":
        clip_image_encoder = CLIPModel.from_pretrained(
            os.path.join(model_root, image_kwargs.get("image_encoder_subpath", "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth")),
        ).eval()
        clip_image_encoder.requires_grad_(False)

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    return WanT2VBundle(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        vae=vae,
        transformer=transformer,
        scheduler=scheduler,
        clip_image_encoder=clip_image_encoder,
    )
