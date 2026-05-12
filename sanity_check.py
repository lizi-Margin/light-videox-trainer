from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import Accelerator

from data.datamodule import VideoDataModule
from models.wan_loader import load_wan_t2v_bundle
from utils.config import load_jsonc
from utils.device import dtype_from_mixed_precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Check config, metadata, and optionally model loading.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--load-model", action="store_true")
    args = parser.parse_args()

    cfg = load_jsonc(args.config)
    data_module = VideoDataModule.from_config(cfg)
    accelerator = Accelerator()

    print(f"Config: {Path(args.config).resolve()}")
    print(f"Metadata: {data_module.config.metadata_path}")
    print(f"Dataset size: {len(data_module.train_dataset)}")
    batch = next(iter(data_module.train_dataloader(accelerator)))
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
