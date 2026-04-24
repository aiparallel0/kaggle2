"""Append-only scalar tracker backing the ``curves/`` CSVs + ``telemetry.jsonl``.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: single :class:`Tracker` the training loops in ``models/*_train.py``
    use to record per-step / per-epoch scalars.  Writes one CSV per
    scalar under ``<run_dir>/curves/<name>.csv`` (``step,value``
    columns, header on first write) and mirrors every record into
    ``<run_dir>/metrics/telemetry.jsonl`` as a JSON line.  Append-only
    so an interrupted run leaves a valid partial file a later
    ``make paper`` invocation can still plot.

The API is intentionally minimal (2-in/1-out):

>>> tr = Tracker(run_dir)
>>> tr.log("donut_loss", step=12, value=0.73)
>>> tr.log("gpu_util", step=12, value=87.4)
>>> tr.close()

Adding a new scalar is a single ``tr.log(name, step, value)`` call;
no CSV / JSONL boilerplate in callers.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import TextIO

log = logging.getLogger("kaggle2")


class Tracker:
    """Append-only per-run scalar tracker.

    Thread-safe for writers that log from multiple threads (e.g. a GPU
    telemetry collector + the main train loop).  Never raises on
    filesystem errors — logs and continues so a flaky disk doesn't
    abort training.
    """

    def __init__(self, run_dir: str | Path) -> None:
        self._run_dir = Path(run_dir)
        self._curves_dir = self._run_dir / "curves"
        self._metrics_dir = self._run_dir / "metrics"
        self._curves_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._csv_handles: dict[str, TextIO] = {}
        self._lock = threading.Lock()
        self._jsonl_path = self._metrics_dir / "telemetry.jsonl"
        self._jsonl_handle: TextIO | None = None

    def _open_csv(self, name: str) -> TextIO | None:
        handle = self._csv_handles.get(name)
        if handle is not None:
            return handle
        path = self._curves_dir / f"{name}.csv"
        new_file = not path.exists() or path.stat().st_size == 0
        try:
            handle = path.open("a", encoding="utf-8")
        except OSError as exc:
            log.warning("tracker: cannot open %s (%s); dropping scalar", path, exc)
            return None
        if new_file:
            handle.write("step,value,wall_time\n")
            handle.flush()
        self._csv_handles[name] = handle
        return handle

    def _open_jsonl(self) -> TextIO | None:
        if self._jsonl_handle is not None:
            return self._jsonl_handle
        try:
            self._jsonl_handle = self._jsonl_path.open("a", encoding="utf-8")
        except OSError as exc:
            log.warning("tracker: cannot open %s (%s); dropping jsonl", self._jsonl_path, exc)
            return None
        return self._jsonl_handle

    def log(self, name: str, step: int, value: float) -> None:
        """Append ``(step, value)`` to ``<run_dir>/curves/<name>.csv``."""
        wall = time.time()
        with self._lock:
            csv = self._open_csv(name)
            if csv is not None:
                try:
                    csv.write(f"{step},{value!r},{wall!r}\n")
                    csv.flush()
                except OSError as exc:
                    log.warning("tracker: write to %s failed (%s)", name, exc)
            jsonl = self._open_jsonl()
            if jsonl is not None:
                try:
                    jsonl.write(json.dumps({
                        "name": name, "step": step, "value": value, "wall_time": wall,
                    }) + "\n")
                    jsonl.flush()
                except OSError as exc:
                    log.warning("tracker: jsonl write failed (%s)", exc)

    def log_many(self, step: int, scalars: dict[str, float]) -> None:
        """Convenience: record every entry of ``scalars`` at ``step``."""
        for name, value in scalars.items():
            self.log(name, step, float(value))

    def close(self) -> None:
        """Close every opened CSV + JSONL handle.  Idempotent."""
        import contextlib
        with self._lock:
            for handle in self._csv_handles.values():
                with contextlib.suppress(OSError):
                    handle.close()
            self._csv_handles.clear()
            if self._jsonl_handle is not None:
                with contextlib.suppress(OSError):
                    self._jsonl_handle.close()
                self._jsonl_handle = None

    def __enter__(self) -> Tracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
