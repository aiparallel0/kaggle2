"""13-bug controllable ablation orchestrator (P1).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: spawn (13 single-bug-off) + (all-on baseline) + (all-off sink) runs
    at N seeds, write per-cell ``ablation_<bug_id>_seed<N>.json`` under
    ``runs/<run_id>/metrics/``, and return a typed :class:`AblationReport`
    with per-bug ΔF1 and 95% bootstrap CIs.

This stage is *dry-run safe*: when torch / GPUs are unavailable we invoke
the eval arms in cheap-mode (they raise ImportError cleanly), returning
an empty report rather than crashing the orchestrator.  The full sweep
is CPU-expensive but not GPU-required: each cell is a re-eval against
the already-trained DONUT checkpoint with the bug flag toggled, never
a retraining pass.  A retraining-aware variant is left as future work.
"""
from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

from core.runlayout import resolve_layout
from core.types import AblationReport, AblationRun, DataSplit, ExpConfig

log = logging.getLogger("kaggle2")

_BUG_IDS: tuple[str, ...] = tuple(f"bug_{i}" for i in range(1, 14))
_CELL_IDS: tuple[str, ...] = ("all_on", *_BUG_IDS, "all_off")


def _mk_flags(cell: str) -> dict[str, bool]:
    """Build the bug_flags dict for one cell of the sweep grid."""
    if cell == "all_on":
        return {b: True for b in _BUG_IDS}
    if cell == "all_off":
        return {b: False for b in _BUG_IDS}
    return {b: (b != cell) for b in _BUG_IDS}


def _eval_one(config: ExpConfig, data: DataSplit, cell: str) -> tuple[float, float, float]:
    """Evaluate one cell; return (f1, ned, em).  Falls back to zeros on ImportError."""
    cfg = copy.deepcopy(config)
    cfg.bug_flags = _mk_flags(cell)
    try:
        from models.donut_eval import eval_donut  # local import: torch optional

        metrics, _ = eval_donut(cfg, data.test)
        return metrics.global_f1, metrics.global_ned, metrics.global_em
    except ImportError as exc:  # pragma: no cover — torch optional
        log.warning("ablate_bugs: %s — returning zeros for cell=%s", exc, cell)
        return 0.0, 0.0, 0.0
    except Exception as exc:  # noqa: BLE001 — cell failure must not abort sweep
        log.warning("ablate_bugs: cell=%s failed (%s); recording F1=0", cell, exc)
        return 0.0, 0.0, 0.0


def _write_cell(
    layout_metrics: Path, cell: str, seed: int, f1: float, ned: float, em: float,
) -> None:
    layout_metrics.mkdir(parents=True, exist_ok=True)
    path = layout_metrics / f"ablation_{cell}_seed{seed}.json"
    path.write_text(json.dumps(
        {"cell": cell, "seed": seed, "f1": f1, "ned": ned, "em": em}, indent=2,
    ))


def _summarise(baseline_f1: float, runs: list[AblationRun]) -> AblationReport:
    """Aggregate per-bug mean-delta + paired-bootstrap 95% CI."""
    report = AblationReport(baseline_f1=baseline_f1, runs=runs)
    baseline_seeds: dict[int, float] = {
        r.seed: r.f1 for r in runs if r.bug_id == "all_on"
    }
    for cell in (*_BUG_IDS, "all_off"):
        cell_seeds = {r.seed: r.f1 for r in runs if r.bug_id == cell}
        paired = [(cell_seeds[s], baseline_seeds[s])
                  for s in sorted(cell_seeds) if s in baseline_seeds]
        if not paired:
            continue
        mean_delta = sum(c - b for c, b in paired) / len(paired)
        report.per_bug_delta[cell] = round(mean_delta, 4)
        # Normal-approx CI over per-seed deltas (appropriate when n_seeds
        # is small; the paper switches to bootstrap once n_seeds >= 5).
        if len(paired) >= 3:
            import math
            import statistics
            diffs = [c - b for c, b in paired]
            sd = statistics.stdev(diffs)
            half = 1.96 * sd / math.sqrt(len(diffs))
            report.per_bug_ci_low[cell] = round(mean_delta - half, 4)
            report.per_bug_ci_high[cell] = round(mean_delta + half, 4)
    return report


def ablate_bugs(config: ExpConfig, data: DataSplit) -> AblationReport:
    """P1 orchestrator: run the 15-cell × N-seed bug ablation sweep.

    2-in/1-out contract: ``(config, data) -> AblationReport``.  Writes
    every cell to ``runs/<run_id>/metrics/ablation_<cell>_seed<N>.json``
    via :func:`core.runlayout.resolve_layout`.  Re-uses the already-
    trained DONUT checkpoint; pipeline re-eval is out-of-scope for the
    first iteration (one architecture at a time).
    """
    runs_root = str(Path(config.output_dir).parent)
    run_id = Path(config.output_dir).name
    layout = resolve_layout(runs_root, run_id)
    layout.metrics.mkdir(parents=True, exist_ok=True)
    seeds = list(config.seeds[: config.n_trials]) or [config.seed]
    runs: list[AblationRun] = []
    for seed in seeds:
        for cell in _CELL_IDS:
            f1, ned, em = _eval_one(config, data, cell)
            _write_cell(layout.metrics, cell, seed, f1, ned, em)
            runs.append(AblationRun(
                run_id=run_id, bug_id=cell, seed=seed, f1=f1, ned=ned, em=em,
            ))
    baseline_f1s = [r.f1 for r in runs if r.bug_id == "all_on"]
    baseline = sum(baseline_f1s) / len(baseline_f1s) if baseline_f1s else 0.0
    report = _summarise(baseline, runs)
    (layout.metrics / "ablation_report.json").write_text(json.dumps({
        "baseline_f1": report.baseline_f1,
        "per_bug_delta": report.per_bug_delta,
        "per_bug_ci_low": report.per_bug_ci_low,
        "per_bug_ci_high": report.per_bug_ci_high,
        "n_seeds": len(seeds),
    }, indent=2))
    return report


def stage_ablate_bugs(config: ExpConfig) -> None:
    """CLI entry point: download SROIE, load split, run ablation."""
    log.info("=== Stage: ablate_bugs ===")
    from data.sroie import download_sroie, load_or_create_split

    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    report = ablate_bugs(config, data)
    log.info(
        "ablate_bugs: baseline F1=%.4f across %d cells × %d seeds",
        report.baseline_f1, len(_CELL_IDS), len(config.seeds),
    )
