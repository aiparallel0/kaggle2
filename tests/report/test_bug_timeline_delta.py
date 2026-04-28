"""Audit B2 regression: per-bug ΔF1 is computed as ``ceiling - f1_before``.

* For every bug in ``results/bug_timeline.json``, the emitted
  ``bug_<N>_delta`` lies in ``[-1, +1]``.
* ``|ΔF1| ≤ |ceiling - floor| + 1e-6``.
* ``bug_7`` (val/test leakage) is an over-reporting bug, so its
  ΔF1 is strictly negative.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from report.combine_new import merge_ablation_report

_TIMELINE = Path(__file__).resolve().parents[2] / "results" / "bug_timeline.json"
_FIXTURE = Path(__file__).parent.parent / "fixtures" / "canonical_347_n1.json"


def _emit_deltas() -> dict[str, object]:
    """Reproduce the heal path: load fixture metrics + bug_timeline, merge."""
    metrics: dict[str, object] = json.loads(_FIXTURE.read_text())
    config = MagicMock()
    config.output_dir = "/nonexistent"  # forces _emit_from_bug_timeline path
    merge_ablation_report(config, metrics)
    return metrics


def test_every_bug_delta_in_unit_interval() -> None:
    metrics = _emit_deltas()
    timeline: dict[str, object] = json.loads(_TIMELINE.read_text())
    bugs = timeline.get("bugs") or []
    assert isinstance(bugs, list) and bugs
    for entry in bugs:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        if not isinstance(idx, int):
            continue
        d = metrics.get(f"bug_{idx}_delta")
        assert isinstance(d, int | float), f"bug_{idx}_delta missing"
        assert -1.0 <= float(d) <= 1.0


def test_delta_bounded_by_ceiling_minus_floor() -> None:
    metrics = _emit_deltas()
    timeline: dict[str, object] = json.loads(_TIMELINE.read_text())
    ceiling = float(metrics.get("ablation_baseline_f1") or 0.0)
    floor = 0.0
    bound = abs(ceiling - floor) + 1e-6
    bugs = timeline.get("bugs") or []
    for entry in bugs:
        if not isinstance(entry, dict):
            continue
        idx = entry.get("id")
        if not isinstance(idx, int):
            continue
        d = metrics.get(f"bug_{idx}_delta")
        assert isinstance(d, int | float)
        assert abs(float(d)) <= bound, f"bug_{idx} delta={d} exceeds ceiling-floor={bound}"


def test_bug_7_delta_is_negative() -> None:
    """val/test leakage was over-reporting → post-fix ceiling < pre-fix F1."""
    metrics = _emit_deltas()
    d = metrics.get("bug_7_delta")
    assert isinstance(d, int | float)
    assert float(d) < 0.0, (
        f"bug_7 (val/test leakage) was over-reporting, expected negative ΔF1; got {d}"
    )


def test_bugs_10_and_13_deltas_are_consistent() -> None:
    """bugs 10 & 13 fixed real failures → their ΔF1 must be positive."""
    metrics = _emit_deltas()
    for bug_id in (10, 13):
        d = metrics.get(f"bug_{bug_id}_delta")
        assert isinstance(d, int | float)
        # Positive only when pre-fix F1 < ceiling — both these bugs fit
        # the canonical-fixture's ceiling (0.8216).
        assert float(d) > 0.0, f"bug_{bug_id} expected positive ΔF1; got {d}"
