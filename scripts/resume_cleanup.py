"""Disk cleanup helpers for the post-TrOCR recovery script.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: wipes DONUT checkpoints, YOLO staging images, and package caches
    to free disk before promoting the best TrOCR checkpoint.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("resume")


def checkpoint_step(path: Path) -> int:
    """Extract integer step from checkpoint-<step> directory names."""
    try:
        return int(path.name.split("-", 1)[1])
    except (IndexError, ValueError):
        return -1


def rmtree_quiet(path: Path) -> int:
    """shutil.rmtree that returns bytes freed and ignores ENOENT."""
    if not path.exists():
        return 0
    try:
        freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    except OSError:
        freed = 0
    shutil.rmtree(path, ignore_errors=True)
    return freed


def cleanup_disk(config: ExpConfig) -> None:
    """Free disk: DONUT per-epoch checkpoints, YOLO staging, pip/apt caches."""
    root = Path(config.output_dir)
    freed = 0
    donut_dir = root / "donut"
    if donut_dir.is_dir():
        for cp in donut_dir.glob("checkpoint-*"):
            if cp.is_dir():
                freed += rmtree_quiet(cp)
    # YOLO staging: a full copy of every SROIE training image next to YOLO's
    # label .txt files. Regenerated deterministically on the next train run.
    freed += rmtree_quiet(root / "yolo_data")
    # pip + apt caches. Tolerate failure — we're root on vast.ai but the
    # script should still make progress on stricter setups.
    subprocess.run(
        ["pip", "cache", "purge"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["apt-get", "clean"],
        check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    log.info("Freed ~%.1f MiB from DONUT checkpoints + yolo_data/.",
             freed / (1024 * 1024))


def print_disk(prefix: str) -> None:
    """Log current disk-free/total in GiB (tolerates OSError)."""
    try:
        usage = shutil.disk_usage(".")
        log.info(
            "%s disk free=%.1f GiB / total=%.1f GiB",
            prefix, usage.free / (1024 ** 3), usage.total / (1024 ** 3),
        )
    except OSError:
        pass
