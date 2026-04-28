"""PR-B — Multi-seed canonical-SROIE aggregator."""
from __future__ import annotations

import json
from pathlib import Path

from conftest import write_min_config


def _write_run(
    root: Path, run_id: str, f1: float, delta: float,
    a_corr: list[bool], b_corr: list[bool],
) -> Path:
    d = root / run_id
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "pipeline_global_f1": f1,
        "delta_f1": delta,
        "donut_per_image_correct": a_corr,
        "pipeline_per_image_correct": b_corr,
    }
    (d / "combined_metrics.json").write_text(json.dumps(payload))
    return d


def test_aggregate_seeds_emits_required_keys(tmp_path: Path) -> None:
    from core.config import load_config
    from report.combine_seeds import aggregate_seeds

    seed_dirs = [
        _write_run(tmp_path, "seed_42", 0.81, 0.01,
                   [True, False, True], [True, True, True]),
        _write_run(tmp_path, "seed_1", 0.79, -0.005,
                   [True, True, False], [True, True, True]),
        _write_run(tmp_path, "seed_2", 0.82, 0.02,
                   [False, True, True], [True, True, True]),
    ]
    cfg = load_config(str(write_min_config(tmp_path)))
    out = aggregate_seeds(cfg, seed_dirs)
    for k in (
        "assigner_n_seeds", "assigner_f1_mean", "assigner_f1_std",
        "delta_f1_mean", "delta_f1_ci_lo", "delta_f1_ci_hi",
        "delta_f1_pvalue_mcnemar_exact",
    ):
        assert k in out, f"missing {k}"
    assert out["assigner_n_seeds"] == 3
    assert 0.79 <= float(out["assigner_f1_mean"]) <= 0.82  # type: ignore[arg-type]


def test_aggregate_seeds_under_threshold_returns_empty(tmp_path: Path) -> None:
    from core.config import load_config
    from report.combine_seeds import aggregate_seeds

    seed_dirs = [_write_run(tmp_path, "only", 0.5, 0.0, [True], [True])]
    cfg = load_config(str(write_min_config(tmp_path)))
    out = aggregate_seeds(cfg, seed_dirs)
    assert out == {}
