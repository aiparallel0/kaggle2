"""FOCUS-C span head — soft no-regression floor on pipeline company F1.

Loads the latest run's ``extended_metrics.json`` (or any pipeline-metrics
JSON the operator has produced) and asserts ``pipeline_f1_company`` is
no worse than ``0.85``.  Skips cleanly when no metrics file is found
so the test passes on a fresh clone with no runs yet.

The hard ``≥ 0.92`` acceptance is validated by the operator on a fresh
GPU run (``python main.py --stage focus`` then ``--stage eval`` then
``--stage paper``).  This CI gate is only the loose floor that catches
silent regressions of the existing pipeline_f1_company contract.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _candidate_metric_paths() -> list[Path]:
    """Possible places an extended/pipeline metrics JSON may live."""
    paths: list[Path] = []
    runs_root = Path("runs")
    if runs_root.is_dir():
        for run in sorted(runs_root.iterdir(), reverse=True):
            if run.is_dir():
                for name in ("extended_metrics.json", "pipeline_metrics.json"):
                    cand = run / name
                    if cand.is_file():
                        paths.append(cand)
    for legacy in (
        Path("results/extended_metrics.json"),
        Path("results/pipeline_metrics.json"),
    ):
        if legacy.is_file():
            paths.append(legacy)
    return paths


def test_pipeline_company_f1_floor() -> None:
    """``pipeline_f1_company`` ≥ 0.85 when the metrics file exists."""
    paths = _candidate_metric_paths()
    if not paths:
        pytest.skip("no extended_metrics.json / pipeline_metrics.json fixture")
    raw = json.loads(paths[0].read_text())
    if not isinstance(raw, dict):
        pytest.skip(f"{paths[0]} is not a JSON object")
    f1 = raw.get("pipeline_f1_company")
    if f1 is None:
        pytest.skip(f"{paths[0]} missing 'pipeline_f1_company'")
    assert float(f1) >= 0.85, (
        f"pipeline_f1_company={f1!r} regressed below the 0.85 CI floor "
        f"(target ≥ 0.92 after FOCUS-C span head + boilerplate anchor)"
    )
