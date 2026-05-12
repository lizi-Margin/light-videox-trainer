from __future__ import annotations

import argparse
from pathlib import Path

from accelerate import Accelerator

from data.datamodule import VideoDataModule
from models.wan_loader import load_wan_t2v_bundle
from trainer.wan_trainer import train_wan
from utils.config import load_jsonc
from utils.device import dtype_from_mixed_precision


def _add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="Path to a .jsonc config file.")


def run_train(args: argparse.Namespace) -> None:
    train_wan(load_jsonc(args.config))


def run_sanity(args: argparse.Namespace) -> None:
    cfg = load_jsonc(args.config)
    data_module = VideoDataModule.from_config(cfg)
    accelerator = Accelerator()

    print(f"Config: {Path(args.config).resolve()}")
    print(f"Task: {cfg.get('task', 't2v')}")
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
        if bundle.clip_image_encoder is not None:
            print(f"Loaded image encoder: {bundle.clip_image_encoder.__class__.__name__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight Wan video trainer.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training.")
    _add_config_arg(train_parser)
    train_parser.set_defaults(func=run_train)

    sanity_parser = subparsers.add_parser("sanity", help="Validate config, data, and optional model loading.")
    _add_config_arg(sanity_parser)
    sanity_parser.add_argument("--load-model", action="store_true", help="Also load model weights.")
    sanity_parser.set_defaults(func=run_sanity)

    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    parsed_args.func(parsed_args)
