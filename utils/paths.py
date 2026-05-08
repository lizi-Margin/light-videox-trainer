from __future__ import annotations

import os
from pathlib import Path


def expand_path(path: str | None, base_dir: str | None = None) -> str | None:
    if path is None or path == "":
        return path
    expanded = Path(os.path.expandvars(os.path.expanduser(path)))
    if not expanded.is_absolute() and base_dir:
        expanded = Path(base_dir) / expanded
    return str(expanded.resolve())


def ensure_dir(path: str | Path) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(Path(path))

