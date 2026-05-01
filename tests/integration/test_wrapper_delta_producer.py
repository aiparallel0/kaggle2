"""``report.wrapper_delta.merge_wrapper_delta`` producer tests.

Project: kaggle2 — FOCUS-$\\Sigma$ verification layer for document KIE.
Role: smoke-test that the NeurIPS-variant wrapper-$\\Delta$ producer
    folds its sidecars into the flat ``\\VAR{}`` dict consumed by
    ``report.inject.inject_results``, and that on a missing sidecar
    no synthetic numbers are fabricated (unresolved keys are left
    unresolved so they surface in
    ``runs/<id>/metrics/unresolved_vars.json``).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace


def _cfg(out_dir: Path, **extra: object) -> SimpleNamespace:
    """Minimal ExpConfig stub — only attrs the producer reads."""
    base: dict[str, object] = {
        "output_dir": str(out_dir),
        "cord_test_n": None,
        "layoutlmv3_epochs": None,
    }
    base.update(extra)
    return SimpleNamespace(**base)


def _write(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_wrapper_delta_forwards_scalar_keys(tmp_path: Path) -> None:
    from report.wrapper_delta import merge_wrapper_delta

    _write(
        tmp_path / "metrics" / "wrapper_delta_metrics.json",
        {
            "schema_version": 1,
            "donut_total_bare": 0.812,
            "donut_total_focus_sigma": 0.851,
            "layoutlmv3_total_bare": 0.857,
            "layoutlmv3_total_focus_sigma": 0.881,
            "pipeline_f1_total_focus_sigma": 0.93,
            "pipeline_f1_total_focus_sigma_ci_half": 0.012,
        },
    )
    out: dict[str, object] = {}
    merge_wrapper_delta(_cfg(tmp_path), out)

    assert out["donut_total_bare"] == 0.812
    assert out["donut_total_focus_sigma"] == 0.851
    assert out["layoutlmv3_total_bare"] == 0.857
    assert out["pipeline_f1_total_focus_sigma"] == 0.93
    assert out["pipeline_f1_total_focus_sigma_ci_half"] == 0.012
    # ``schema_version`` is metadata, not a paper key.
    assert "schema_version" not in out


def test_wrapper_delta_renders_ci_lists_as_strings(tmp_path: Path) -> None:
    from report.wrapper_delta import merge_wrapper_delta

    _write(
        tmp_path / "metrics" / "wrapper_delta_metrics.json",
        {
            "donut_total_focus_sigma_delta_ci": [0.018, 0.061],
            "layoutlmv3_total_focus_sigma_delta_ci": [0.012, 0.039],
        },
    )
    out: dict[str, object] = {}
    merge_wrapper_delta(_cfg(tmp_path), out)
    assert out["donut_total_focus_sigma_delta_ci"] == "[0.0180, 0.0610]"
    assert out["layoutlmv3_total_focus_sigma_delta_ci"] == "[0.0120, 0.0390]"


def test_wrapper_delta_no_sidecar_no_fabrication(tmp_path: Path) -> None:
    """Honest gap: missing sidecar leaves keys unresolved (not zero)."""
    from report.wrapper_delta import merge_wrapper_delta

    out: dict[str, object] = {}
    merge_wrapper_delta(_cfg(tmp_path), out)
    # No measurement keys are populated.
    for k in (
        "donut_total_bare",
        "donut_total_focus_sigma",
        "layoutlmv3_total_focus_sigma",
        "pipeline_f1_total_focus_sigma",
        "cord_macro_focus_sigma",
        "ablation_row3_total_f1",
        "faithfulness_deletion_auc",
        "calibration_ece_post",
        "lat_verifier_p50",
    ):
        assert k not in out, (
            f"producer fabricated {k!r} when no sidecar was present"
        )


def test_wrapper_delta_heals_baseline_from_headline(tmp_path: Path) -> None:
    """``pipeline_f1_total_baseline`` is forwarded from the headline pipeline_f1_total."""
    from report.wrapper_delta import merge_wrapper_delta

    out: dict[str, object] = {"pipeline_f1_total": 0.72}
    merge_wrapper_delta(_cfg(tmp_path), out)
    assert out["pipeline_f1_total_baseline"] == 0.72
    # The post-Σ key MUST NOT be synthesised from the bare key.
    assert "pipeline_f1_total_focus_sigma" not in out


def test_wrapper_delta_config_provenance_keys(tmp_path: Path) -> None:
    from report.wrapper_delta import merge_wrapper_delta

    out: dict[str, object] = {}
    merge_wrapper_delta(
        _cfg(tmp_path, cord_test_n=100, layoutlmv3_epochs=8), out,
    )
    assert out["cord_test_n"] == 100
    assert out["layoutlmv3_epochs"] == 8


def test_wrapper_delta_setdefault_does_not_override(tmp_path: Path) -> None:
    """An explicit producer-written value wins over the sidecar."""
    from report.wrapper_delta import merge_wrapper_delta

    _write(
        tmp_path / "metrics" / "wrapper_delta_metrics.json",
        {"donut_total_bare": 0.812},
    )
    out: dict[str, object] = {"donut_total_bare": 0.999}
    merge_wrapper_delta(_cfg(tmp_path), out)
    assert out["donut_total_bare"] == 0.999  # explicit wins


def test_wrapper_delta_aggregates_all_sidecars(tmp_path: Path) -> None:
    from report.wrapper_delta import merge_wrapper_delta

    _write(
        tmp_path / "metrics" / "ablation_focus_sigma.json",
        {"ablation_row3_total_f1": 0.911, "ablation_row3_delta_point": 0.041},
    )
    _write(
        tmp_path / "metrics" / "faithfulness_metrics.json",
        {"faithfulness_deletion_auc": 0.18, "faithfulness_insertion_auc": 0.71},
    )
    _write(
        tmp_path / "metrics" / "calibration_metrics.json",
        {"calibration_ece_pre": 0.12, "calibration_ece_post": 0.04},
    )
    _write(
        tmp_path / "metrics" / "latency_metrics.json",
        {"lat_verifier_p50": 32.4, "lat_verifier_p95": 39.1},
    )
    out: dict[str, object] = {}
    merge_wrapper_delta(_cfg(tmp_path), out)
    assert out["ablation_row3_total_f1"] == 0.911
    assert out["faithfulness_insertion_auc"] == 0.71
    assert out["calibration_ece_post"] == 0.04
    assert out["lat_verifier_p50"] == 32.4
