from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def read_caption_table(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]]
    if p.suffix.lower() == ".json":
        payload = json.loads(p.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else list(payload.values())
    else:
        with p.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    table: dict[str, str] = {}
    for row in rows:
        file_value = (
            row.get("file_path")
            or row.get("path")
            or row.get("video")
            or row.get("video_path")
            or row.get("filename")
            or row.get("id")
        )
        text = row.get("text") or row.get("caption") or row.get("prompt") or row.get("short_prompt") or ""
        if not file_value:
            continue
        file_name = Path(str(file_value)).name
        table[file_name] = str(text)
        table[Path(file_name).stem] = str(text)
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description="Create VideoX-style metadata JSON from OpenVid/video folders.")
    parser.add_argument("--video-root", required=True, help="Folder containing extracted videos.")
    parser.add_argument("--caption-table", default=None, help="Optional CSV/JSON with filename/path and caption/text columns.")
    parser.add_argument("--output", required=True, help="Output metadata JSON.")
    parser.add_argument("--relative-to", default=None, help="Store paths relative to this directory.")
    parser.add_argument("--default-caption", default="", help="Caption used when no table entry is found.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of videos.")
    args = parser.parse_args()

    video_root = Path(args.video_root).resolve()
    rel_root = Path(args.relative_to).resolve() if args.relative_to else None
    captions = read_caption_table(args.caption_table)
    videos = sorted(p for p in video_root.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if args.limit > 0:
        videos = videos[: args.limit]

    items = []
    missing_caption = 0
    for video in videos:
        text = captions.get(video.name) or captions.get(video.stem) or args.default_caption
        if not text:
            missing_caption += 1
        file_path = str(video.relative_to(rel_root)) if rel_root else str(video)
        items.append({"file_path": file_path, "text": text, "type": "video"})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(items)} items to {output}")
    if missing_caption:
        print(f"Items without captions: {missing_caption}")


if __name__ == "__main__":
    main()

