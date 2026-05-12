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

    @classmethod
    def from_dict(cls, cfg: dict[str, Any]) -> "VideoDataConfig":
        config_dir = cfg.get("_config_dir")
        return cls(
            metadata_path=expand_path(cfg["train_metadata"], config_dir),
            data_root=expand_path(cfg.get("train_data_root") or None, config_dir),
            sample_n_frames=int(cfg["video_sample_n_frames"]),
            sample_stride=int(cfg["video_sample_stride"]),
            sample_size=sample_size_from_config(cfg["video_sample_size"]),
            text_drop_ratio=float(cfg.get("text_drop_ratio", 0.0)),
            random_crop=bool(cfg.get("random_crop", True)),
            batch_size=int(cfg["train_batch_size"]),
            num_workers=int(cfg.get("dataloader_num_workers", 0)),
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
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            persistent_workers=bool(self.config.num_workers),
            collate_fn=collator,
            drop_last=True,
        )
