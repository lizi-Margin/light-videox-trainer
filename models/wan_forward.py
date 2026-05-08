from __future__ import annotations

import math

import torch
from einops import rearrange


def encode_video_latents(vae, pixel_values: torch.Tensor, mini_batch: int = 1) -> torch.Tensor:
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    encoded = []
    for start in range(0, pixel_values.shape[0], mini_batch):
        chunk = pixel_values[start : start + mini_batch]
        latent = vae.encode(chunk)[0].sample()
        encoded.append(latent)
    return torch.cat(encoded, dim=0)


def encode_prompts(tokenizer, text_encoder, texts: list[str], device: torch.device, max_length: int) -> list[torch.Tensor]:
    prompt_ids = tokenizer(
        texts,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    text_input_ids = prompt_ids.input_ids.to(device)
    attention_mask = prompt_ids.attention_mask.to(device)
    prompt_embeds = text_encoder(text_input_ids, attention_mask=attention_mask)[0]
    seq_lens = attention_mask.gt(0).sum(dim=1).long()
    return [embed[:seq_len] for embed, seq_len in zip(prompt_embeds, seq_lens)]


def transformer_seq_len(transformer, vae, latents: torch.Tensor) -> int:
    _, _, num_frames, height, width = latents.shape
    patch_t, patch_h, patch_w = transformer.config.patch_size
    _ = patch_t
    return math.ceil((width * height) / (patch_h * patch_w) * num_frames)

