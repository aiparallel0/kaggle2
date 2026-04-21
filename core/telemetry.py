"""Lightweight GPU + system telemetry sampler for training runs.

Polls nvidia-smi (subprocess) once per ``interval_s`` and writes one JSONL
line per tick.  Falls back to a ``{"note": "no-gpu"}`` line and exits
cleanly when nvidia-smi is absent.  CPU / RAM / disk are sampled via
``psutil`` when available.

Public API (2-in / 1-out per function):
  start_sampler(out_path, interval_s)  -> (Thread, Event)
  stop_sampler(thread, stop_event)     -> str (out_path)
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
    """Return GPU metrics dict from nvidia-smi, or {} on any failure."""
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
    """Return CPU / RAM / disk metrics dict, or {} if psutil is absent."""
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
    """Background sampling loop; writes JSONL to out_path until stop_event."""
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
    """Start a background telemetry-sampling thread.

    Args:
        out_path: Path for JSONL output (parent dirs created automatically).
        interval_s: Sampling interval in seconds (default 5).

    Returns:
        ``(thread, stop_event)`` — pass both to :func:`stop_sampler`.
    """
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
    """Stop the telemetry thread and return the output path.

    Args:
        thread: Thread returned by :func:`start_sampler`.
        stop_event: Event returned by :func:`start_sampler`.

    Returns:
        The JSONL output path (stored as thread.name).
    """
    stop_event.set()
    thread.join(timeout=15.0)
    return str(thread.name)
