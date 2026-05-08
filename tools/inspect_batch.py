from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader

from data.collate import VideoCollator
from data.video_dataset import VideoCaptionDataset
from utils.config import load_jsonc
from utils.paths import expand_path
from utils.video_io import save_video_preview


def _sample_size(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Save a decoded/cropped training batch preview.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="outputs/inspect_batch.mp4")
    args = parser.parse_args()

    cfg = load_jsonc(args.config)
    metadata = expand_path(cfg["train_metadata"], cfg.get("_config_dir"))
    data_root = expand_path(cfg.get("train_data_root") or None, cfg.get("_config_dir"))
    sample_size = _sample_size(cfg["video_sample_size"])
    dataset = VideoCaptionDataset(
        metadata_path=metadata,
        data_root=data_root,
        sample_n_frames=int(cfg["video_sample_n_frames"]),
        sample_stride=int(cfg["video_sample_stride"]),
        sample_size=sample_size,
        text_drop_ratio=0.0,
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=VideoCollator(sample_size))))
    output = Path(args.output)
    save_video_preview(batch["pixel_values"][0], output)
    print(f"Saved {output}")
    print(f"Text: {batch['text'][0]}")
    print(f"Path: {batch['file_path'][0]}")


if __name__ == "__main__":
    main()
