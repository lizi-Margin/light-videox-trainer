from __future__ import annotations

import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from data.collate import VideoCollator
from data.video_dataset import VideoCaptionDataset
from models.wan_loader import load_wan_t2v_bundle
from utils.config import load_jsonc
from utils.device import dtype_from_mixed_precision
from utils.paths import expand_path


def _sample_size(value) -> tuple[int, int]:
    if isinstance(value, int):
        return value, value
    return int(value[0]), int(value[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Check config, metadata, and optionally model loading.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    cfg = load_jsonc(args.config)
    metadata = expand_path(cfg["train_metadata"], cfg.get("_config_dir"))
    data_root = expand_path(cfg.get("train_data_root") or None, cfg.get("_config_dir"))
    sample_size = _sample_size(cfg["video_sample_size"])

    print(f"Config: {Path(args.config).resolve()}")
    print(f"Metadata: {metadata}")
    dataset = VideoCaptionDataset(
        metadata_path=metadata,
        data_root=data_root,
        sample_n_frames=int(cfg["video_sample_n_frames"]),
        sample_stride=int(cfg["video_sample_stride"]),
        sample_size=sample_size,
        text_drop_ratio=float(cfg.get("text_drop_ratio", 0.0)),
    )
    print(f"Dataset size: {len(dataset)}")
    batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=VideoCollator(sample_size))))
    print(f"Batch video shape: {tuple(batch['pixel_values'].shape)}")
    print(f"Batch text: {batch['text'][0]}")
    print(f"Batch path: {batch['file_path'][0]}")

    if args.load_model:
        bundle = load_wan_t2v_bundle(cfg, dtype_from_mixed_precision(cfg.get("mixed_precision", "bf16")))
        print(f"Loaded transformer: {bundle.transformer.__class__.__name__}")
        print(f"Loaded VAE: {bundle.vae.__class__.__name__}")
        print(f"Loaded text encoder: {bundle.text_encoder.__class__.__name__}")


if __name__ == "__main__":
    main()

