"""PR-E — Pareto sweep stage (assigner_size × dataset × seed).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: orchestrate the 24-cell sweep (4 sizes × 2 datasets × 3 seeds) —
    DONUT-base + LayoutLMv3-{base,large} reference rows are reused
    across cells.  Pre-registered hypotheses live in
    ``PRE_REGISTRATION.md`` at repo root; results are aggregated to
    ``runs/_sweep_aggregate.json`` for :mod:`report.figures_pareto_full`.

Invocation: ``python -m stages.sweep --config configs/sweep/<size>.json``
or via the orchestrator: ``python main.py --stage sweep``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("kaggle2")

_SIZES = ("tiny", "small", "base", "large")
_DATASETS = ("sroie", "cord")


def run_sweep(config: ExpConfig) -> Path:
    """Execute the Pareto sweep grid; return the aggregate JSON path.

    Skips cells that have not been trained yet (``runs/<id>/
    combined_metrics.json`` missing) so the sweep can be resumed
    incrementally on a single GPU.  The aggregate dumps every cell's
    ``f1_mean`` / ``f1_std`` / ``params_m`` so the Pareto figure can
    render without re-reading the per-cell JSONs.
    """
    rows: list[dict[str, object]] = []
    for size in _SIZES:
        for ds in _DATASETS:
            for seed in config.seeds:
                cell = _read_cell(config, size, ds, seed)
                if cell is not None:
                    rows.append(cell)
    out = Path("runs") / "_sweep_aggregate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump({"rows": rows}, fh, indent=2, sort_keys=True)
    log.info("stages.sweep: aggregated %d cells -> %s", len(rows), out)
    return out


def _read_cell(
    config: ExpConfig, size: str, dataset: str, seed: int,
) -> dict[str, object] | None:
    """Read one sweep cell's ``combined_metrics.json``; return None on miss."""
    run_id = f"sweep_{size}_{dataset}_seed{seed}"
    path = Path("runs") / run_id / "combined_metrics.json"
    if not path.is_file():
        return None
    try:
        with path.open() as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("stages.sweep: cannot read %s (%s)", path, exc)
        return None
    f1 = data.get("pipeline_global_f1") or data.get("global_f1")
    params = data.get("pipeline_params_m")
    if not (isinstance(f1, int | float) and isinstance(params, int | float)):
        return None
    _ = config  # silence unused-arg warning while keeping the 2-in/1-out shape
    return {
        "size": size,
        "dataset": dataset,
        "seed": seed,
        "f1_mean": float(f1),
        "params_m": float(params),
    }
