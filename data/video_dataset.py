from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from decord import VideoReader
from torch.utils.data import Dataset


def _resolve_file(path: str, data_root: str | None) -> str:
    p = Path(path)
    if not p.is_absolute() and data_root:
        p = Path(data_root) / p
    return str(p.resolve())


def resize_larger_side(frame: np.ndarray, larger_side: int) -> np.ndarray:
    h, w = frame.shape[:2]
    scale = larger_side / max(h, w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


class VideoCaptionDataset(Dataset):
    def __init__(
        self,
        metadata_path: str,
        data_root: str | None = None,
        sample_n_frames: int = 81,
        sample_stride: int = 2,
        sample_size: tuple[int, int] = (256, 256),
        text_drop_ratio: float = 0.0,
        max_items: int = 0,
        require_text: bool = False,
        min_frames: int = 0,
        min_width: int = 0,
        min_height: int = 0,
        min_duration: float = 0.0,
        max_duration: float = 0.0,
        decord_num_threads: int = 2,
    ) -> None:
        self.metadata_path = metadata_path
        self.data_root = data_root
        self.sample_n_frames = sample_n_frames
        self.sample_stride = sample_stride
        self.sample_size = sample_size
        self.text_drop_ratio = text_drop_ratio
        self.decord_num_threads = decord_num_threads

        with open(metadata_path, "r", encoding="utf-8") as f:
            raw_items: list[dict[str, Any]] = json.load(f)
        self.items = [
            item
            for item in raw_items
            if item.get("type", "video") == "video"
            and self._passes_metadata_filters(
                item,
                require_text=require_text,
                min_frames=min_frames,
                min_width=min_width,
                min_height=min_height,
                min_duration=min_duration,
                max_duration=max_duration,
            )
        ]
        if max_items > 0:
            self.items = self.items[:max_items]
        if not self.items:
            raise ValueError(f"No video items found in metadata: {metadata_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[idx % len(self.items)]
        video_path = _resolve_file(item["file_path"], self.data_root)
        text = item.get("text", "")
        if random.random() < self.text_drop_ratio:
            text = ""

        vr = VideoReader(video_path, num_threads=self.decord_num_threads)
        if len(vr) < 1:
            raise ValueError(f"No frames in video: {video_path}")

        needed = (self.sample_n_frames - 1) * self.sample_stride + 1
        if len(vr) >= needed:
            start = random.randint(0, len(vr) - needed)
            indices = np.arange(start, start + needed, self.sample_stride, dtype=np.int64)
        else:
            indices = np.linspace(0, len(vr) - 1, self.sample_n_frames, dtype=np.int64)

        frames = vr.get_batch(indices).asnumpy()
        larger_side = max(self.sample_size)
        frames = np.stack([resize_larger_side(frame, larger_side) for frame in frames], axis=0)
        return {
            "pixel_values": frames,
            "text": text,
            "file_path": video_path,
            "data_type": "video",
        }

    def _passes_metadata_filters(
        self,
        item: dict[str, Any],
        require_text: bool,
        min_frames: int,
        min_width: int,
        min_height: int,
        min_duration: float,
        max_duration: float,
    ) -> bool:
        if require_text and not str(item.get("text", "")).strip():
            return False
        frame_count = self._number(item, "num_frames", "frames", "frame_count")
        if min_frames > 0 and frame_count > 0 and frame_count < min_frames:
            return False
        width = self._number(item, "width", "w")
        height = self._number(item, "height", "h")
        if min_width > 0 and width > 0 and width < min_width:
            return False
        if min_height > 0 and height > 0 and height < min_height:
            return False
        duration = self._number(item, "duration", "duration_sec", "seconds")
        if min_duration > 0 and duration > 0 and duration < min_duration:
            return False
        if max_duration > 0 and duration > 0 and duration > max_duration:
            return False
        return True

    def _number(self, item: dict[str, Any], *keys: str) -> float:
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return 0.0
