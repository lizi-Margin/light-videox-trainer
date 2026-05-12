from __future__ import annotations

import os
from typing import Any

from utils.paths import ensure_dir


class UHTKMetricsLogger:
    def __init__(self, cfg: dict[str, Any], output_dir: str, enabled: bool = True) -> None:
        self.enabled = enabled
        self._failed = False
        self._mcv = None
        self._manager = None
        if not self.enabled:
            return

        logdir = os.path.join(output_dir, "visualizer")
        ensure_dir(logdir)

        try:
            from uhtk.mcv_log_manager import LogManager, get_a_logger

            self._mcv = get_a_logger(
                logdir,
                color=str(cfg.get("visualizer_color", "k")),
                figsize=cfg.get("visualizer_figsize"),
                dpi=int(cfg.get("visualizer_dpi", 120)),
                font_size=int(cfg.get("visualizer_font_size", 9)),
            )
            self._manager = LogManager(
                self._mcv,
                who=cfg.get("tracker_project_name", "light-videox-trainer"),
                enable_smooth=bool(cfg.get("visualizer_smooth", True)),
            )
        except Exception as exc:
            self._mark_failed(exc)

    def log(self, metrics: dict[str, float], step: int) -> None:
        if not self.enabled or self._failed or self._manager is None or self._mcv is None:
            return
        try:
            self._mcv.rec(step, "time")
            self._manager.log_trivial(metrics)
            self._manager.log_trivial_finalize(print=True)
        except Exception as exc:
            self._mark_failed(exc)

    def close(self) -> None:
        if self._mcv is None:
            return
        try:
            self._mcv.rec_end()
        except Exception:
            pass

    def _mark_failed(self, exc: Exception) -> None:
        if not self._failed:
            print(f"[UHTKMetricsLogger] disabled after logging error: {exc}")
        self._failed = True
