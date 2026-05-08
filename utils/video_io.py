from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def tensor_to_video_uint8(video: torch.Tensor) -> np.ndarray:
    """Convert [F, C, H, W] in [-1, 1] or [0, 1] to uint8 [F, H, W, C]."""
    video = video.detach().float().cpu()
    if video.min() < 0:
        video = (video + 1.0) / 2.0
    video = video.clamp(0, 1)
    return (video.permute(0, 2, 3, 1).numpy() * 255).round().astype(np.uint8)


def save_video_preview(video: torch.Tensor, path: str | Path, fps: int = 8) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frames = tensor_to_video_uint8(video)
    try:
        import mediapy as media

        media.write_video(str(path), frames, fps=fps)
        return
    except Exception:
        pass

    import cv2

    h, w = frames.shape[1:3]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
