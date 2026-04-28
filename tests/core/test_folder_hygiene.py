"""Tests for the runs/<run_id>/ folder-hygiene invariants.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: guard the contract in ``core/runlayout.py``.  ``results/`` must
    stay fixtures-only; ``runs/`` must be ignored wholesale except for
    the tracked README; ``derive_paths`` must produce the
    ``runs/<run_id>/paper/paper_filled.tex`` shape when ``runs_root``
    is set.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

from core.runlayout import (
    SUBDIRS,
    derive_paths,
    ensure_subdirs,
    latest_run,
    make_run_id,
    resolve_layout,
)

REPO = Path(__file__).resolve().parents[2]


def test_results_is_fixtures_only() -> None:
    """Only the shipped fixtures may live under ``results/``."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "results"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    allowed = {
        "results/bug_timeline.json",
        "results/foundation_baseline.json",
        "results/README.md",
        "results/sroie_task3_competitors.json",
    }
    assert set(tracked) <= allowed, f"unexpected tracked files: {set(tracked) - allowed}"


def test_runs_dir_is_ignored_except_readme() -> None:
    """``runs/`` must be git-ignored wholesale with only README tracked."""
    tracked = subprocess.run(
        ["git", "-C", str(REPO), "ls-files", "runs"],
        capture_output=True, text=True, check=True,
    ).stdout.splitlines()
    assert tracked == ["runs/README.md"], tracked


def test_make_run_id_shape() -> None:
    """run_id must be ``<16-char-UTC>-<7-char-sha>`` (or nosha fallback)."""
    rid = make_run_id(REPO)
    ts, _, sha = rid.partition("-")
    assert len(ts) == 16 and ts.endswith("Z"), rid
    assert sha and (len(sha) == 7 or sha == "nosha"), rid


def test_resolve_layout_has_every_subdir(tmp_path: Path) -> None:
    """Every name in SUBDIRS gets a concrete Path attribute on RunLayout."""
    layout = resolve_layout(tmp_path, "20260101T000000Z-deadbee")
    for name in SUBDIRS:
        sub = getattr(layout, name)
        assert sub.name == name
        assert sub.parent == layout.run_dir
    ensure_subdirs(layout)
    for name in SUBDIRS:
        assert (layout.run_dir / name).is_dir()


def test_derive_paths_back_compat(tmp_path: Path) -> None:
    """Without ``runs_root``, config's raw paths must survive unchanged."""
    out, paper = derive_paths(
        cfg_output_dir="./results",
        cfg_paper_output="./report/paper_filled.tex",
        cfg_runs_root=None,
        cfg_run_id=None,
        repo_root=tmp_path,
    )
    assert out == "./results"
    assert paper == "./report/paper_filled.tex"


def test_derive_paths_under_runs_root(tmp_path: Path) -> None:
    """With ``runs_root`` set, paper goes under runs/<run_id>/paper/."""
    out, paper = derive_paths(
        cfg_output_dir="./results",
        cfg_paper_output="./report/paper_filled.tex",
        cfg_runs_root=str(tmp_path / "runs"),
        cfg_run_id="20260101T000000Z-deadbee",
        repo_root=tmp_path,
    )
    assert out.endswith("/runs/20260101T000000Z-deadbee")
    assert paper.endswith("/runs/20260101T000000Z-deadbee/paper/paper_filled.tex")


def test_latest_run_picks_newest(tmp_path: Path) -> None:
    """``latest_run`` returns the directory with the most recent mtime."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    (runs_root / "older").mkdir()
    newer = runs_root / "newer"
    newer.mkdir()
    # Force the newer directory to have a later mtime regardless of OS
    # filesystem timestamp resolution.
    os.utime(newer, (10**9, 10**9))
    os.utime(runs_root / "older", (10**9 - 60, 10**9 - 60))
    assert latest_run(runs_root) == newer
