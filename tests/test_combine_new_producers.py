"""Producers in report.combine_new fold P1/P2/P4 side-cars into \\VAR{}.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: smoke-test that the three new merge helpers
    (:func:`merge_ablation_report`, :func:`merge_foundation_metrics`,
    :func:`merge_rag_metrics`) surface their side-car JSONs into the
    flat metrics dict that :func:`report.inject.inject_results`
    consumes.  Uses a temp-dir layout (no network, no torch).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _cfg(out_dir: Path) -> SimpleNamespace:
    """Minimal ExpConfig-shaped stub: only ``output_dir`` is read."""
    return SimpleNamespace(output_dir=str(out_dir))


def _write(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_merge_ablation_report_emits_bug_delta_and_ci(tmp_path: Path) -> None:
    from report.combine_new import merge_ablation_report

    _write(
        tmp_path / "metrics" / "ablation_report.json",
        {
            "baseline_f1": 0.82,
            "n_seeds": 3,
            "per_bug_delta": {"bug_1": -0.30, "all_off": -0.55},
            "per_bug_ci_low": {"bug_1": -0.34, "all_off": -0.60},
            "per_bug_ci_high": {"bug_1": -0.26, "all_off": -0.50},
        },
    )
    out: dict[str, object] = {}
    merge_ablation_report(_cfg(tmp_path), out)
    assert out["ablation_baseline_f1"] == 0.82
    assert out["ablation_n_seeds"] == 3
    assert out["bug_1_delta"] == -0.30
    assert out["bug_1_ci_low"] == -0.34
    assert out["bug_1_ci_high"] == -0.26
    assert out["all_off_delta"] == -0.55


def test_merge_foundation_metrics_prefixes_correctly(tmp_path: Path) -> None:
    from report.combine_new import merge_foundation_metrics

    _write(
        tmp_path / "metrics" / "foundation_metrics.json",
        {
            "global_f1": 0.91, "global_ned": 0.06, "global_em": 0.72,
            "per_field_f1": {"company": 0.95, "total": 0.88},
        },
    )
    out: dict[str, object] = {}
    merge_foundation_metrics(_cfg(tmp_path), out)
    assert out["foundation_f1"] == 0.91
    assert out["foundation_ned"] == 0.06
    assert out["foundation_em"] == 0.72
    assert out["foundation_f1_company"] == 0.95
    assert out["foundation_f1_total"] == 0.88


def test_merge_rag_metrics_forwards_keys_verbatim(tmp_path: Path) -> None:
    from report.combine_new import merge_rag_metrics

    _write(
        tmp_path / "metrics" / "rag_ablation.json",
        {
            "schema_version": 1,  # dropped on fold
            "rag_on_f1": 0.87, "rag_off_f1": 0.81,
            "rag_on_ned": 0.09, "rag_off_ned": 0.14,
        },
    )
    out: dict[str, object] = {}
    merge_rag_metrics(_cfg(tmp_path), out)
    assert out["rag_on_f1"] == 0.87
    assert out["rag_off_f1"] == 0.81
    assert out["rag_on_ned"] == 0.09
    assert "schema_version" not in out


def test_missing_sidecars_are_silent(tmp_path: Path) -> None:
    """Every helper must no-op when the side-car is absent."""
    from report.combine_new import (
        merge_ablation_report,
        merge_foundation_metrics,
        merge_rag_metrics,
    )
    cfg = _cfg(tmp_path)
    out: dict[str, object] = {}
    merge_ablation_report(cfg, out)
    merge_foundation_metrics(cfg, out)
    merge_rag_metrics(cfg, out)
    assert out == {}
