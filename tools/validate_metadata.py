from __future__ import annotations

import argparse
import json
from pathlib import Path

from decord import VideoReader
from tqdm import tqdm


def resolve(path: str, root: str | None) -> Path:
    p = Path(path)
    if not p.is_absolute() and root:
        p = Path(root) / p
    return p.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate video metadata paths and frame readability.")
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--write-valid", default=None)
    args = parser.parse_args()

    items = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    if args.limit > 0:
        items = items[: args.limit]

    valid = []
    bad = 0
    for item in tqdm(items):
        path = resolve(item["file_path"], args.data_root)
        try:
            vr = VideoReader(str(path), num_threads=1)
            if len(vr) < args.min_frames:
                raise ValueError(f"only {len(vr)} frames")
            _ = vr[0]
            valid.append(item)
        except Exception as exc:
            bad += 1
            print(f"BAD {path}: {exc}")

    print(f"Valid: {len(valid)} / {len(items)}")
    print(f"Bad: {bad}")
    if args.write_valid:
        out = Path(args.write_valid)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(valid, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote valid metadata: {out}")


if __name__ == "__main__":
    main()

