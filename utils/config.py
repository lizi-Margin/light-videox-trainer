from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _strip_json_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    lines = []
    for line in text.splitlines():
        in_string = False
        escaped = False
        keep = []
        i = 0
        while i < len(line):
            ch = line[i]
            nxt = line[i + 1] if i + 1 < len(line) else ""
            if ch == "\\" and in_string:
                escaped = not escaped
                keep.append(ch)
                i += 1
                continue
            if ch == '"' and not escaped:
                in_string = not in_string
            escaped = False
            if not in_string and ch == "/" and nxt == "/":
                break
            keep.append(ch)
            i += 1
        lines.append("".join(keep))
    return "\n".join(lines)


def load_jsonc(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    raw = config_path.read_text(encoding="utf-8")
    try:
        import commentjson

        cfg = commentjson.loads(raw)
    except Exception:
        cfg = json.loads(_strip_json_comments(raw))
    cfg["_config_path"] = str(config_path.resolve())
    cfg["_config_dir"] = str(config_path.resolve().parent)
    return cfg


def save_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def get_nested(cfg: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

