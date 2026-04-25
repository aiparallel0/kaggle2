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
    """Minimal ExpConfig-shaped stub: only ``output_dir`` and ``rag_k`` read."""
    return SimpleNamespace(output_dir=str(out_dir), rag_k=3)


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


def test_missing_sidecars_use_repo_fixtures(tmp_path: Path) -> None:
    """Heal contract: when no live side-car is present, the mergers
    fall back to the repo-tracked fixtures so every \\VAR{} resolves.

    * ``merge_ablation_report`` heals from ``results/bug_timeline.json``
      (asserted in :func:`test_merge_ablation_report_falls_back_to_bug_timeline`).
    * ``merge_foundation_metrics`` heals from
      ``results/foundation_baseline.json``.
    * ``merge_rag_metrics`` heals the RAG-off row from the already-
      present headline ``donut_*`` keys + ``config.rag_k``.
    """
    from report.combine_new import merge_foundation_metrics, merge_rag_metrics
    cfg = _cfg(tmp_path)
    out: dict[str, object] = {
        "donut_f1": 0.84, "donut_ned": 0.92, "donut_em": 0.75,
    }
    merge_foundation_metrics(cfg, out)
    merge_rag_metrics(cfg, out)
    assert "foundation_f1" in out and "foundation_ned" in out
    assert out["rag_off_f1"] == 0.84
    assert out["rag_k"] == int(cfg.rag_k)


def test_merge_ablation_report_falls_back_to_bug_timeline(tmp_path: Path) -> None:
    """No ablation_report.json + repo fixture present → keys synthesised."""
    from report.combine_new import merge_ablation_report
    out: dict[str, object] = {}
    merge_ablation_report(_cfg(tmp_path), out)
    # 13 bugs × 3 keys + all_off × 3 + ablation_baseline_f1 + ablation_n_seeds
    assert "bug_1_delta" in out and "bug_13_delta" in out
    assert "all_off_delta" in out
    assert out["ablation_n_seeds"] == 1
