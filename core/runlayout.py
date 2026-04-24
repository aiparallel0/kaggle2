"""Single source of truth for the per-run output-directory layout.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: resolve ``runs_root`` + ``run_id`` → concrete ``Path``s for every
    sub-directory (metrics/, curves/, predictions/, attention/, figures/,
    paper/, env/) and provide helpers to create them, migrate legacy
    ``./results/`` layouts, and enumerate files for ``MANIFEST.json``.
    Every writer in stages/ and models/ MUST route through this module —
    never hard-code a path string under ``runs/``.
"""
from __future__ import annotations

import datetime as _dt
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Subdirectory names are the reviewer-facing contract (mirrored in the
# ``results/`` README and the paper's reproducibility appendix).  Never
# rename these without updating docs/TRACKING.md.
SUBDIRS: tuple[str, ...] = (
    "metrics",
    "curves",
    "predictions",
    "attention",
    "figures",
    "paper",
    "env",
)


@dataclass(frozen=True)
class RunLayout:
    """Concrete paths for one run.  All absolute (or runs_root-relative)."""

    run_id: str
    run_dir: Path
    metrics: Path
    curves: Path
    predictions: Path
    attention: Path
    figures: Path
    paper: Path
    env: Path


def _git_short_sha(repo_root: Path) -> str:
    """Return the 7-char short SHA of HEAD; ``nosha`` if git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short=7", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nosha"
    return proc.stdout.strip() or "nosha"


def make_run_id(repo_root: Path, now: _dt.datetime | None = None) -> str:
    """``<UTC>-<gitsha>`` run identifier, e.g. ``20260424T103055Z-a1b2c3d``."""
    ts = (now or _dt.datetime.now(_dt.UTC)).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}-{_git_short_sha(repo_root)}"


def resolve_layout(runs_root: str | os.PathLike[str], run_id: str) -> RunLayout:
    """Build a :class:`RunLayout` for ``run_id`` under ``runs_root``."""
    root = Path(runs_root).resolve()
    run_dir = root / run_id
    return RunLayout(
        run_id=run_id,
        run_dir=run_dir,
        metrics=run_dir / "metrics",
        curves=run_dir / "curves",
        predictions=run_dir / "predictions",
        attention=run_dir / "attention",
        figures=run_dir / "figures",
        paper=run_dir / "paper",
        env=run_dir / "env",
    )


def ensure_subdirs(layout: RunLayout) -> None:
    """Create every sub-directory in ``layout`` (idempotent)."""
    layout.run_dir.mkdir(parents=True, exist_ok=True)
    for name in SUBDIRS:
        (layout.run_dir / name).mkdir(parents=True, exist_ok=True)


def latest_run(runs_root: str | os.PathLike[str]) -> Path | None:
    """Return the most-recently-modified run directory, or ``None``."""
    root = Path(runs_root)
    if not root.is_dir():
        return None
    candidates = [p for p in root.iterdir() if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def derive_paths(
    cfg_output_dir: str,
    cfg_paper_output: str,
    cfg_runs_root: str | None,
    cfg_run_id: str | None,
    repo_root: Path,
) -> tuple[str, str]:
    """Derive (output_dir, paper_output) from config + env.

    When ``runs_root`` is falsy we return the raw config values — this
    is the back-compat path still exercised by existing tests.  When
    ``runs_root`` is set (in config.json or via ``KAGGLE2_RUNS_ROOT``)
    we build ``<runs_root>/<run_id>/`` and place the paper under
    ``paper/paper_filled.tex``.  ``run_id`` may be pinned via
    ``KAGGLE2_RUN_ID`` or ``config.run_id`` so ``main.py`` /
    ``scripts/eval_only.py`` can target a previous run.
    """
    runs_root = os.environ.get("KAGGLE2_RUNS_ROOT") or cfg_runs_root
    run_id = os.environ.get("KAGGLE2_RUN_ID") or cfg_run_id
    if not runs_root:
        return cfg_output_dir, cfg_paper_output
    rid = str(run_id) if run_id else make_run_id(repo_root)
    layout = resolve_layout(runs_root, rid)
    ensure_subdirs(layout)
    return str(layout.run_dir), str(layout.paper / "paper_filled.tex")
