from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from accelerate import Accelerator
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

from data.collate import VideoCollator
from data.video_dataset import VideoCaptionDataset
from utils.paths import expand_path

logger = get_logger(__name__)


def sample_size_from_config(value: Any) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    if isinstance(value, list) and len(value) == 2:
        return int(value[0]), int(value[1])
    if isinstance(value, tuple) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid video_sample_size: {value}")


@dataclass
class VideoDataConfig:
    metadata_path: str
    data_root: str | None
    sample_n_frames: int
    sample_stride: int
    sample_size: tuple[int, int]
    text_drop_ratio: float
    random_crop: bool
    batch_size: int
    num_workers: int
    max_items: int
    require_text: bool
    min_frames: int
    min_width: int
    min_height: int
    min_duration: float
    max_duration: float
    decord_num_threads: int
    pin_memory: bool
    prefetch_factor: int | None

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VideoDataConfig":
        config_dir = cfg.get("_config_dir")
        sample_n_frames = int(cfg["video_sample_n_frames"])
        sample_stride = int(cfg["video_sample_stride"])
        return cls(
            metadata_path=expand_path(cfg["train_metadata"], config_dir),
            data_root=expand_path(cfg.get("train_data_root") or None, config_dir),
            sample_n_frames=sample_n_frames,
            sample_stride=sample_stride,
            sample_size=sample_size_from_config(cfg["video_sample_size"]),
            text_drop_ratio=float(cfg.get("text_drop_ratio", 0.0)),
            random_crop=bool(cfg.get("random_crop", True)),
            batch_size=int(cfg["train_batch_size"]),
            num_workers=int(cfg.get("dataloader_num_workers", 0)),
            max_items=int(cfg.get("data_max_items", 0)),
            require_text=bool(cfg.get("data_require_text", False)),
            min_frames=int(cfg.get("data_min_frames", 0)),
            min_width=int(cfg.get("data_min_width", 0)),
            min_height=int(cfg.get("data_min_height", 0)),
            min_duration=float(cfg.get("data_min_duration", 0.0)),
            max_duration=float(cfg.get("data_max_duration", 0.0)),
            decord_num_threads=int(cfg.get("decord_num_threads", 2)),
            pin_memory=bool(cfg.get("dataloader_pin_memory", True)),
            prefetch_factor=cfg.get("dataloader_prefetch_factor"),
        )


class VideoDataModule:
    def __init__(self, config: VideoDataConfig) -> None:
        self.config = config
        self._train_dataset: VideoCaptionDataset | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "VideoDataModule":
        return cls(VideoDataConfig.from_dict(cfg))

    @property
    def train_dataset(self) -> VideoCaptionDataset:
        if self._train_dataset is None:
            self._train_dataset = VideoCaptionDataset(
                metadata_path=self.config.metadata_path,
                data_root=self.config.data_root,
                sample_n_frames=self.config.sample_n_frames,
                sample_stride=self.config.sample_stride,
                sample_size=self.config.sample_size,
                text_drop_ratio=self.config.text_drop_ratio,
                max_items=self.config.max_items,
                require_text=self.config.require_text,
                min_frames=self.config.min_frames,
                min_width=self.config.min_width,
                min_height=self.config.min_height,
                min_duration=self.config.min_duration,
                max_duration=self.config.max_duration,
                decord_num_threads=self.config.decord_num_threads,
            )
        return self._train_dataset

    def train_dataloader(self, accelerator: Accelerator) -> DataLoader:
        dataset = self.train_dataset
        if accelerator.is_main_process:
            logger.info("Dataset size: %s", len(dataset))
        collator = VideoCollator(
            sample_size=self.config.sample_size,
            random_crop=self.config.random_crop,
        )
        loader_kwargs: dict[str, Any] = {
            "batch_size": self.config.batch_size,
            "shuffle": True,
            "num_workers": self.config.num_workers,
            "persistent_workers": bool(self.config.num_workers),
            "collate_fn": collator,
            "drop_last": True,
            "pin_memory": self.config.pin_memory,
        }
        if self.config.num_workers > 0 and self.config.prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = int(self.config.prefetch_factor)
        return DataLoader(dataset, **loader_kwargs)
