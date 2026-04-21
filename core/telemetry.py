"""Background GPU/CPU telemetry sampler for the paper's efficiency analysis.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: polls nvidia-smi at configurable intervals, writing JSONL rows that
    drive fig_gpu_telemetry and the cost/energy columns in Table II.
    Falls back gracefully on CPU-only runs.  2-in/1-out contract.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from threading import Event, Thread

try:
    import psutil as _psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

_NVIDIA_FIELDS = (
    "utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu"
)


def _query_gpu() -> dict[str, object]:
    """Return GPU util/mem/power/temp from nvidia-smi, or {} on failure."""
    if not shutil.which("nvidia-smi"):
        return {}
    cmd = [
        "nvidia-smi",
        f"--query-gpu={_NVIDIA_FIELDS}",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.check_output(cmd, timeout=5, text=True).strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) < 5:
            return {}
        return {
            "gpu_util_pct": float(parts[0]),
            "gpu_mem_used_mb": float(parts[1]),
            "gpu_mem_total_mb": float(parts[2]),
            "gpu_power_w": float(parts[3]),
            "gpu_temp_c": float(parts[4]),
        }
    except Exception:  # noqa: BLE001
        return {}


def _query_system() -> dict[str, object]:
    """Return CPU/RAM/disk metrics, or {} if psutil unavailable."""
    if not _HAS_PSUTIL:
        return {}
    try:
        mem = _psutil.virtual_memory()
        disk = _psutil.disk_usage("/")
        return {
            "cpu_pct": _psutil.cpu_percent(interval=None),
            "ram_used_mb": float(mem.used) / 1024.0 / 1024.0,
            "disk_used_gb": float(disk.used) / 1024.0 / 1024.0 / 1024.0,
        }
    except Exception:  # noqa: BLE001
        return {}


def _run_loop(out_path: str, interval_s: float, stop_event: Event) -> None:
    """Daemon loop: poll GPU/system, append JSONL until stop_event."""
    if not shutil.which("nvidia-smi"):
        with open(out_path, "a") as fh:
            fh.write(json.dumps({"ts": time.time(), "note": "no-gpu"}) + "\n")
        return
    while not stop_event.is_set():
        row: dict[str, object] = {"ts": time.time()}
        row.update(_query_gpu())
        row.update(_query_system())
        with open(out_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        stop_event.wait(interval_s)


def start_sampler(out_path: str, interval_s: float = 5.0) -> tuple[Thread, Event]:
    """Start background telemetry thread; returns (thread, stop_event)."""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    stop_event = Event()
    thread = Thread(
        target=_run_loop,
        args=(out_path, interval_s, stop_event),
        daemon=True,
        name=out_path,  # name encodes the path for stop_sampler
    )
    thread.start()
    return thread, stop_event


def stop_sampler(thread: Thread, stop_event: Event) -> str:
    """Stop telemetry thread; return the JSONL path it wrote."""
    stop_event.set()
    thread.join(timeout=15.0)
    return str(thread.name)
