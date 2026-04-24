"""Shared helpers + rsync filter lists for the fetch sub-command.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: defines rsync include/exclude patterns to pull metrics without
    transferring heavy model weights unless --with-weights is passed.
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import shlex
import subprocess
import time

log = logging.getLogger("inspect")

_FETCH_INCLUDES = [
    "results/",
    "results/*.json",
    "results/yolo/",
    "results/yolo/run/",
    "results/yolo/run/**",
    "results/donut/",
    "results/donut/*.json",
    "results/donut/generation_config.json",
    "results/donut/config.json",
    "results/donut/preprocessor_config.json",
    "results/donut/tokenizer*",
    "results/donut/special_tokens_map.json",
    "results/donut/added_tokens.json",
    "results/trocr/",
    "results/trocr/*.json",
    "results/trocr/preprocessor_config.json",
    "results/trocr/tokenizer*",
    "results/trocr/vocab.json",
    "results/trocr/merges.txt",
    "results/trocr/special_tokens_map.json",
    "results/assigner.pt",
    "report/",
    "report/paper_filled.tex",
    "report/paper_filled.pdf",
]

_FETCH_EXCLUDES_BASE = [
    "results/yolo_data/",           # staging mirror — huge, regenerated
    "results/yolo/run/weights/last.pt",
    "results/**/checkpoint-*",      # per-epoch checkpoints — training artefacts
]

_FETCH_WEIGHT_PATTERNS = [
    "results/donut/model.safetensors",
    "results/donut/pytorch_model.bin",
    "results/trocr/model.safetensors",
    "results/trocr/pytorch_model.bin",
    "results/yolo/run/weights/best.pt",
]


def build_ssh_cmd(args: argparse.Namespace) -> tuple[list[str], str]:
    """Parse --ssh-cmd / (--port, target) → (ssh-binary-args, host)."""
    if args.ssh_cmd:
        parts = shlex.split(args.ssh_cmd)
        if not parts:
            raise SystemExit("--ssh-cmd must not be empty")
        host = parts[-1]
        ssh_binary = parts[:-1] if len(parts) > 1 else ["ssh"]
        return ssh_binary, host
    if not args.target:
        raise SystemExit(
            "fetch: provide user@host (and optionally -p PORT) or --ssh-cmd '...'",
        )
    ssh_binary = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if args.port:
        ssh_binary += ["-p", str(args.port)]
    return ssh_binary, args.target


def run_with_retry(cmd: list[str], retries: int) -> None:
    """Run cmd with exponential-backoff retries (rsync network resilience)."""
    delay = 2.0
    attempts = 0
    while True:
        attempts += 1
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError as exc:
            if attempts > retries:
                raise SystemExit(
                    f"Command failed after {attempts} attempts: {exc}",
                ) from exc
            log.warning(
                "Attempt %d/%d failed (exit %d) — retrying in %.0fs",
                attempts, retries + 1, exc.returncode, delay,
            )
            time.sleep(delay)
            delay *= 2


def silent_call(cmd: list[str]) -> None:
    with contextlib.suppress(OSError):
        subprocess.run(cmd, check=False, capture_output=True)


def capture(cmd: list[str]) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except OSError as exc:
        return f"<error: {exc}>\n"
    return (result.stdout or "") + (result.stderr or "")


def rsync_filters(with_weights: bool) -> list[str]:
    """Build rsync --include/--exclude args; final * denies everything else."""
    includes = list(_FETCH_INCLUDES)
    excludes = list(_FETCH_EXCLUDES_BASE)
    if with_weights:
        includes += _FETCH_WEIGHT_PATTERNS
    else:
        excludes += [
            "results/donut/model.safetensors",
            "results/donut/pytorch_model.bin",
            "results/trocr/model.safetensors",
            "results/trocr/pytorch_model.bin",
            "results/yolo/run/weights/*.pt",
        ]
    excludes.append("*")
    out: list[str] = []
    for p in includes:
        out += [f"--include={p}"]
    for p in excludes:
        out += [f"--exclude={p}"]
    return out


def remote_env(ssh_binary: list[str], host: str, remote_root: str) -> str:
    """Capture brief remote env snapshot (date, GPU, git HEAD, ls results)."""
    env_script = (
        "echo == date ==; date -u;"
        "echo == uname ==; uname -a;"
        f"echo == df ==; df -h {remote_root} || true;"
        "echo == nvidia-smi ==; nvidia-smi || true;"
        "echo == python ==; python --version 2>&1;"
        "echo == torch ==; python -c "
        "\"import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())\" "
        "2>&1 || true;"
        f"echo == git HEAD ==; (cd {remote_root} && git rev-parse HEAD && git status --porcelain);"
        f"echo == ls results ==; ls -la {remote_root}/results 2>&1 | head -200;"
    )
    return capture([*ssh_binary, host, "bash", "-lc", env_script])
