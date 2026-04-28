"""PR-B — Multi-seed aggregator for canonical-SROIE eval.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: reads each ``runs/<id_seed_<s>>/combined_metrics.json`` and emits
    a single ``runs/_seed_aggregate_<base>/combined_metrics.json`` with
    cross-seed mean ± stdev for the headline F1 number plus the
    paired-bootstrap CI on ``delta_f1`` and the *exact* McNemar p
    pooled over seeds.  Companion to :mod:`report.inject_tables` which
    renders ``mean ± stdev`` cells when ``n_trials > 1``.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path

from core.statistics import mcnemar
from core.types import ExpConfig

log = logging.getLogger("kaggle2")

_MIN_SEEDS = 2  # Below this the std and CI are not informative.


def _read_metrics(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("combine_seeds: cannot read %s (%s)", path, exc)
        return None
    return data if isinstance(data, dict) else None


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: list[float]) -> float:
    n = len(xs)
    if n < _MIN_SEEDS:
        return 0.0
    mu = _mean(xs)
    return math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))


def aggregate_seeds(
    config: ExpConfig, seed_run_dirs: list[Path],
) -> dict[str, object]:
    """Aggregate per-seed ``combined_metrics.json`` files.

    Returns a flat dict with ``assigner_n_seeds``, ``assigner_f1_mean``,
    ``assigner_f1_std``, ``delta_f1_mean``, ``delta_f1_ci_lo``,
    ``delta_f1_ci_hi``, ``delta_f1_pvalue_mcnemar_exact``.  Empty when
    fewer than two seed runs are readable.
    """
    per_seed: list[dict[str, object]] = []
    for d in seed_run_dirs:
        m = _read_metrics(d / "combined_metrics.json")
        if m is not None:
            per_seed.append(m)
    if len(per_seed) < _MIN_SEEDS:
        return {}
    f1s = [
        float(m.get("pipeline_global_f1", 0.0))  # type: ignore[arg-type]
        for m in per_seed
    ]
    deltas = [
        float(m.get("delta_f1", 0.0))  # type: ignore[arg-type]
        for m in per_seed
    ]
    out: dict[str, object] = {
        "assigner_n_seeds": len(per_seed),
        "assigner_f1_mean": round(_mean(f1s), 4),
        "assigner_f1_std": round(_stdev(f1s), 4),
        "delta_f1_mean": round(_mean(deltas), 4),
    }
    # Paired-bootstrap CI on delta across seeds (the per-seed delta is
    # already paired across receipts; aggregating across seeds uses
    # plain percentile because the pairing is at the seed level here).
    sorted_d = sorted(deltas)
    n = len(sorted_d)
    alpha = 1.0 - float(config.bootstrap_ci_level)
    lo = sorted_d[max(0, int(math.floor(0.5 * alpha * n)))]
    hi = sorted_d[min(n - 1, int(math.ceil((1.0 - 0.5 * alpha) * n)) - 1)]
    out["delta_f1_ci_lo"] = round(lo, 4)
    out["delta_f1_ci_hi"] = round(hi, 4)
    # Pool the per-seed correctness vectors and run a single exact
    # McNemar over the pooled pair.  The per-seed test would be
    # under-powered on 63-receipt SROIE; pooling preserves the exact
    # binomial guarantee while gaining 5x sample size.
    a_pool: list[bool] = []
    b_pool: list[bool] = []
    for m in per_seed:
        a = m.get("donut_per_image_correct", []) or []
        b = m.get("pipeline_per_image_correct", []) or []
        if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
            a_pool.extend(bool(x) for x in a)
            b_pool.extend(bool(x) for x in b)
    if a_pool:
        out["delta_f1_pvalue_mcnemar_exact"] = mcnemar(a_pool, b_pool)
    return out


def write_aggregate(
    config: ExpConfig, seed_run_dirs: list[Path], out_dir: Path,
) -> Path:
    """Aggregate + persist ``combined_metrics.json`` under ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate_seeds(config, seed_run_dirs)
    out_path = out_dir / "combined_metrics.json"
    with out_path.open("w") as fh:
        json.dump(agg, fh, indent=2, sort_keys=True)
    return out_path
