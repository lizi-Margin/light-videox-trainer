from __future__ import annotations

from typing import Any

from trainer.wan_trainer import train_wan


def train(cfg: dict[str, Any]) -> None:
    train_wan(cfg)
