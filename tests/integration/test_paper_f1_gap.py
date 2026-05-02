"""``report.paper_f1_gap`` tests — Paper 2 vs Paper 3 F1 gap producer.

Project: kaggle2 — FOCUS-$\\Sigma$ verification layer for document KIE.
Role: smoke-test that the gap reporter (a) reads ``pipeline_f1`` from
    both run dirs' ``combined_metrics.json``, (b) writes
    ``paper_f1_gap.json`` with the three keys consumed by the LaTeX
    layer, and (c) returns an empty dict and writes nothing when
    either run is incomplete (no fabricated numbers).
"""
from __future__ import annotations

import json
from pathlib import Path


def _write_combined(run_dir: Path, pipeline_f1: float) -> None:
    metrics = run_dir / "metrics"
    metrics.mkdir(parents=True, exist_ok=True)
    (metrics / "combined_metrics.json").write_text(
        json.dumps({"pipeline_f1": pipeline_f1}),
    )


def test_compute_paper_f1_gap_reports_three_keys(tmp_path: Path) -> None:
    from report.paper_f1_gap import compute_paper_f1_gap

    p2 = tmp_path / "paper2_run"
    p3 = tmp_path / "paper3_run"
    _write_combined(p2, 0.802)
    _write_combined(p3, 0.913)
    out = compute_paper_f1_gap(p2, p3)
    assert out["paper2_pipeline_f1"] == 0.802
    assert out["paper3_pipeline_f1"] == 0.913
    assert out["paper2_paper3_f1_gap"] == 0.111


def test_compute_paper_f1_gap_returns_empty_on_missing_run(tmp_path: Path) -> None:
    from report.paper_f1_gap import compute_paper_f1_gap

    p2 = tmp_path / "paper2_run"
    p3 = tmp_path / "paper3_run"
    _write_combined(p2, 0.78)
    # paper3_run does not exist
    out = compute_paper_f1_gap(p2, p3)
    assert out == {}


def test_compute_paper_f1_gap_returns_empty_on_missing_key(tmp_path: Path) -> None:
    from report.paper_f1_gap import compute_paper_f1_gap

    p2 = tmp_path / "paper2_run"
    p3 = tmp_path / "paper3_run"
    p2_metrics = p2 / "metrics"
    p2_metrics.mkdir(parents=True)
    (p2_metrics / "combined_metrics.json").write_text(
        json.dumps({"donut_f1": 0.85}),  # no pipeline_f1
    )
    _write_combined(p3, 0.91)
    assert compute_paper_f1_gap(p2, p3) == {}


def test_write_paper_f1_gap_persists_to_paper3_metrics(tmp_path: Path) -> None:
    from report.paper_f1_gap import write_paper_f1_gap

    p2 = tmp_path / "paper2_run"
    p3 = tmp_path / "paper3_run"
    _write_combined(p2, 0.79)
    _write_combined(p3, 0.92)
    out_path = write_paper_f1_gap(p2, p3)
    assert out_path is not None
    payload = json.loads(out_path.read_text())
    assert payload["paper2_paper3_f1_gap"] == 0.13
    assert out_path == p3 / "metrics" / "paper_f1_gap.json"


def test_write_paper_f1_gap_skips_when_runs_incomplete(tmp_path: Path) -> None:
    """Honest gap: no synthetic write when one run is missing."""
    from report.paper_f1_gap import write_paper_f1_gap

    p2 = tmp_path / "paper2_run"
    p3 = tmp_path / "paper3_run"
    _write_combined(p2, 0.80)
    out_path = write_paper_f1_gap(p2, p3)
    assert out_path is None
    # The paper3 metrics dir is created (mkdir parents=True) but no
    # file is written — the existence check is on the file, not the dir.
    assert not (p3 / "metrics" / "paper_f1_gap.json").exists()
