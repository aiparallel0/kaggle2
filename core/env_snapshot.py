"""Write the ``<run_dir>/env/`` reproducibility snapshot.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: capture exactly-one-per-run artefacts that let reviewers
    reconstruct the environment: git short SHA, ``pip freeze`` output,
    ``nvidia-smi`` dump, the effective ``config.json`` at run start,
    and a ``hostinfo.json`` with CPU / RAM / GPU / CUDA / driver /
    torch / python details.  Every writer is best-effort — failures
    are logged but never raise so a missing ``nvidia-smi`` on a CPU
    box doesn't abort the run.
"""
from __future__ import annotations

import hashlib
import json
import logging
import platform
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from core.runlayout import _git_short_sha
from core.schemas import SCHEMA_VERSIONS, EnvSnapshot, HostInfo

log = logging.getLogger("kaggle2")


def _safe_run(cmd: list[str], timeout: float = 10.0) -> str:
    """Run ``cmd`` and return stdout; empty string on any failure."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
        log.warning("env_snapshot: %s failed (%s)", cmd[0], exc)
        return ""
    return out.stdout or ""


def _collect_host_info() -> HostInfo:
    info: HostInfo = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    try:
        import os
        info["cpu_count"] = os.cpu_count() or 0
    except OSError:
        pass
    try:
        import torch
        info["torch_version"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_model"] = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            info["gpu_vram_gb"] = round(props.total_memory / (1024 ** 3), 2)
            info["cuda_version"] = torch.version.cuda or ""
    except ImportError:
        pass
    # nvidia-smi provides the driver version even when torch is absent.
    if shutil.which("nvidia-smi"):
        smi = _safe_run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
        if smi:
            info["driver_version"] = smi.strip().splitlines()[0]
    return info


def _config_sha256(config_path: Path) -> str:
    """Stable 64-hex sha256 of the effective config file on disk."""
    if not config_path.is_file():
        return ""
    h = hashlib.sha256(config_path.read_bytes()).hexdigest()
    return h


def write_env_snapshot(
    env_dir: Path,
    config_path: Path,
    run_id: str,
    seed: int,
) -> Path:
    """Write every file under ``env/`` and return the hostinfo.json path."""
    env_dir.mkdir(parents=True, exist_ok=True)
    repo_root = config_path.resolve().parent
    sha = _git_short_sha(repo_root)
    (env_dir / "git_sha.txt").write_text(sha + "\n")
    # pip freeze can be slow; 30-second timeout is generous.
    freeze = _safe_run([sys.executable, "-m", "pip", "freeze"], timeout=30.0)
    (env_dir / "pip_freeze.txt").write_text(freeze)
    smi = _safe_run(["nvidia-smi"])
    (env_dir / "nvidia_smi.txt").write_text(smi)
    # Snapshot the exact config.json we're booting with so post-hoc
    # sweeps can diff two runs with a one-liner.
    if config_path.is_file():
        (env_dir / "config_snapshot.json").write_text(config_path.read_text())
    host = _collect_host_info()
    snap: EnvSnapshot = {
        "schema_version": SCHEMA_VERSIONS["EnvSnapshot"],
        "run_id": run_id,
        "run_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_sha": sha,
        "config_sha256": _config_sha256(config_path),
        "seed": seed,
        "host": host,
    }
    out = env_dir / "hostinfo.json"
    out.write_text(json.dumps(snap, indent=2, sort_keys=False))
    return out
