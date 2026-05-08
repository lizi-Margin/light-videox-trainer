from __future__ import annotations

import argparse

from trainer.wan_t2v_trainer import train
from utils.config import load_jsonc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Wan T2V with a lightweight VideoX-Fun harness.")
    parser.add_argument("--config", required=True, help="Path to a .jsonc config file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(load_jsonc(args.config))

