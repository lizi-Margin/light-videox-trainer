from __future__ import annotations

import os
from pathlib import Path


def latest_checkpoint(output_dir: str) -> str | None:
    path = Path(output_dir)
    if not path.exists():
        return None
    checkpoints = [p for p in path.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.name.split("-")[-1]))
    return str(checkpoints[-1])


def resolve_resume_checkpoint(output_dir: str, resume_from_checkpoint: str | None) -> str | None:
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint == "latest":
        return latest_checkpoint(output_dir)
    if os.path.isabs(resume_from_checkpoint):
        return resume_from_checkpoint
    return os.path.join(output_dir, resume_from_checkpoint)

