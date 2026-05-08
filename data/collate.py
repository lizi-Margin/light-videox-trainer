from __future__ import annotations

import random
from typing import Any

import torch
import torch.nn.functional as F


def _resize_short_side(video: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = video.shape
    scale = max(target_h / h, target_w / w)
    new_h = max(target_h, int(round(h * scale)))
    new_w = max(target_w, int(round(w * scale)))
    return F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)


def _crop(video: torch.Tensor, target_h: int, target_w: int, random_crop: bool) -> torch.Tensor:
    _, _, h, w = video.shape
    if h == target_h and w == target_w:
        return video
    if random_crop:
        top = random.randint(0, max(0, h - target_h))
        left = random.randint(0, max(0, w - target_w))
    else:
        top = max(0, (h - target_h) // 2)
        left = max(0, (w - target_w) // 2)
    return video[:, :, top : top + target_h, left : left + target_w]


class VideoCollator:
    def __init__(self, sample_size: tuple[int, int], random_crop: bool = True) -> None:
        self.sample_size = sample_size
        self.random_crop = random_crop

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        target_h, target_w = self.sample_size
        videos = []
        texts = []
        paths = []
        for example in examples:
            video = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).float() / 255.0
            video = _resize_short_side(video, target_h, target_w)
            video = _crop(video, target_h, target_w, self.random_crop)
            video = (video - 0.5) / 0.5
            videos.append(video)
            texts.append(example["text"])
            paths.append(example["file_path"])
        return {
            "pixel_values": torch.stack(videos, dim=0),
            "text": texts,
            "file_path": paths,
        }

