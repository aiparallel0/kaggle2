"""Evaluation stages: full (DONUT + pipeline + rule-based) and CPU-only.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: runs the three evaluation arms (DONUT, YOLO+TrOCR+Attention,
    rule-based on gold OCR) over one or more seeds, aggregates
    mean/std when multi-seed, and writes the authoritative
    ``combined_metrics.json`` that the paper stage consumes.
"""
from __future__ import annotations

import json
import logging
import math
import os
import statistics
from pathlib import Path

from core.errors import EvalError
from core.seed import seed_everything
from core.statistics import bootstrap_ci, mcnemar, paired_bootstrap_delta_ci
from core.types import DataSplit, ExpConfig, Metrics
from core.validate import validate_f1
from data.sroie import download_sroie, load_or_create_split
from models.donut_eval import eval_donut
from models.pipeline_eval import eval_pipeline
from models.rule_eval import (
    combined_from_rulebased,
    eval_rulebased_gold,
    per_field_injection,
)
from report.combine import build_combined
from stages._common import (
    assert_pipeline_beats_rulebased_gold,
    warn_below_expected,
    warn_pipeline_diagnostics,
)

log = logging.getLogger("kaggle2")


def _eval_donut_or_skip(config: ExpConfig, data: DataSplit) -> Metrics:
    """Eval DONUT iff ``skip_donut`` is False AND a checkpoint exists."""
    donut_model = os.path.join(config.output_dir, "donut")
    if config.skip_donut:
        log.info("skip_donut=True — skipping DONUT eval; reporting zeros.")
        return Metrics(
            global_f1=0.0, global_ned=0.0, global_em=0.0,
            per_field_f1={f: 0.0 for f in config.fields},
            per_field_ned={f: 0.0 for f in config.fields},
            per_field_em={f: 0.0 for f in config.fields},
        )
    if not Path(donut_model).exists():
        raise EvalError(
            f"DONUT checkpoint not found at {donut_model}. Either run train "
            "stage first, or set skip_donut=true for a Phase-1 pipeline-only run.",
        )
    dm = eval_donut(config, data.test)
    validate_f1(dm, "donut")
    warn_below_expected(dm, config, "donut")
    return dm


def _per_seed_metrics(
    config: ExpConfig, data: DataSplit, seed: int,
) -> tuple[Metrics, Metrics, Metrics]:
    """Run one (DONUT, pipeline, rule-based-on-gold) eval triple at `seed`."""
    seed_everything(seed)
    dm = _eval_donut_or_skip(config, data)
    log.info("DONUT F1=%.4f", dm.global_f1)
    pm = eval_pipeline(config, data.test)
    validate_f1(pm.assigner, "pipeline")
    warn_below_expected(pm.assigner, config, "pipeline")
    warn_pipeline_diagnostics(config)
    log.info("Pipeline (assigner)  F1=%.4f", pm.assigner.global_f1)
    log.info("Pipeline (rulebased) F1=%.4f", pm.rulebased.global_f1)
    rb_gold = eval_rulebased_gold(config, data.test)
    log.info("Rule-based (gold OCR) F1=%.4f", rb_gold.global_f1)
    assert_pipeline_beats_rulebased_gold(pm.assigner, rb_gold)
    return dm, pm.assigner, rb_gold  # pm.assigner used by caller; rb too


def _aggregate_seed_variance(
    f1s: list[float], key: str, out: dict[str, object],
) -> None:
    """Emit mean/std/CI across seeds for a single metric key.

    Populates four keys on `out`:
      * `{key}_mean`   — arithmetic mean across seeds
      * `{key}_std`    — sample stdev (0.0 for n=1, an honest "no spread")
      * `{key}_ci_lo` / `{key}_ci_hi` — 95% normal-approx CI across seeds,
        *only* emitted when n >= 2.  With n=1 there is no seed variance
        to confidence-bound and the paper must rely on the bootstrap CI
        over images (see `per_image_bootstrap_*`).
    """
    if not f1s:
        return
    out[f"{key}_mean"] = round(statistics.fmean(f1s), 4)
    if len(f1s) >= 2:
        sd = statistics.stdev(f1s)
        out[f"{key}_std"] = round(sd, 4)
        half = 1.96 * sd / math.sqrt(len(f1s))
        out[f"{key}_ci_lo"] = round(out[f"{key}_mean"] - half, 4)  # type: ignore[operator]
        out[f"{key}_ci_hi"] = round(out[f"{key}_mean"] + half, 4)  # type: ignore[operator]
    else:
        out[f"{key}_std"] = 0.0


def stage_eval(config: ExpConfig, seeds: list[int] | None = None) -> None:
    """Run eval across one or more seeds; aggregate mean/std/CI.

    Keeps the legacy single-seed keys (``donut_f1``, ``pipeline_f1``,
    ``assigner_delta``) populated for back-compat with the paper's
    \\VAR{} substitution.  Multi-seed runs additionally emit
    ``*_mean`` / ``*_std`` / ``*_ci_lo`` / ``*_ci_hi`` so the paper can
    render seed-level uncertainty bands.  For n=1 the paper-side bootstrap
    CI over per-image correctness is still available via
    :func:`core.statistics.bootstrap_ci`.
    """
    log.info("=== Stage: eval ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    run_seeds = list(seeds) if seeds else list(config.seeds[: config.n_trials])
    log.info("Eval harness: n_trials=%d seeds=%s", len(run_seeds), run_seeds)
    donut_f1s: list[float] = []
    pipeline_f1s: list[float] = []
    rulebased_gold_f1s: list[float] = []
    last: dict[str, object] = {}
    for seed in run_seeds:
        if len(run_seeds) > 1:
            log.info("--- Eval seed=%d ---", seed)
        dm = _eval_donut_or_skip(config, data)
        log.info("DONUT F1=%.4f", dm.global_f1)
        seed_everything(seed)
        pm = eval_pipeline(config, data.test)
        validate_f1(pm.assigner, "pipeline")
        warn_below_expected(pm.assigner, config, "pipeline")
        warn_pipeline_diagnostics(config)
        log.info("Pipeline (assigner)  F1=%.4f", pm.assigner.global_f1)
        log.info("Pipeline (rulebased) F1=%.4f", pm.rulebased.global_f1)
        rb_gold = eval_rulebased_gold(config, data.test)
        log.info("Rule-based (gold OCR) F1=%.4f", rb_gold.global_f1)
        assert_pipeline_beats_rulebased_gold(pm.assigner, rb_gold)
        donut_f1s.append(dm.global_f1)
        pipeline_f1s.append(pm.assigner.global_f1)
        rulebased_gold_f1s.append(rb_gold.global_f1)
        last = build_combined(config, dm, pm, rb_gold)
    # Always emit the variance block, even when n=1 — downstream consumers
    # (paper \VAR{}, log dashboards) should not branch on seed count.
    _aggregate_seed_variance(donut_f1s, "donut_f1", last)
    _aggregate_seed_variance(pipeline_f1s, "pipeline_f1", last)
    _aggregate_seed_variance(rulebased_gold_f1s, "rulebased_gold_f1", last)
    # Per-image bootstrap CI + paired McNemar: uses the last run's per-image
    # all-fields-EM correctness vectors (populated by compute_metrics via
    # build_combined). Paired test is valid because DONUT and pipeline are
    # evaluated on the same 63 test images in fixed order.
    d_raw = last.get("donut_per_image_correct")
    p_raw = last.get("pipeline_per_image_correct")
    d_vec = [bool(x) for x in d_raw] if isinstance(d_raw, list) else []
    p_vec = [bool(x) for x in p_raw] if isinstance(p_raw, list) else []
    if p_vec:
        lo, hi = bootstrap_ci(
            p_vec,
            n_iter=config.bootstrap_n_iter,
            ci_level=config.bootstrap_ci_level,
        )
        last["pipeline_bootstrap_ci_lo"] = round(lo, 4)
        last["pipeline_bootstrap_ci_hi"] = round(hi, 4)
    if d_vec and p_vec and len(d_vec) == len(p_vec):
        _, ci_lo, ci_hi = paired_bootstrap_delta_ci(
            d_vec, p_vec,
            n_iter=config.bootstrap_n_iter,
            ci_level=config.bootstrap_ci_level,
        )
        last["delta_f1_ci_lo"] = round(ci_lo, 4)
        last["delta_f1_ci_hi"] = round(ci_hi, 4)
        last["mcnemar_p"] = round(mcnemar(d_vec, p_vec), 4)
    last["seeds_used"] = list(run_seeds)
    last["n_trials"] = len(run_seeds)
    last["bootstrap_n_iter"] = config.bootstrap_n_iter
    last["bootstrap_ci_level"] = config.bootstrap_ci_level
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(last, f, indent=2)


def stage_eval_rulebased_gold(config: ExpConfig) -> None:
    """Rule-based F1 over SROIE gold OCR — no HF / GPU dependency.

    Writes ``results/rulebased_gold_metrics.json`` and a
    ``combined_metrics.json`` pre-populated with zeros for the DONUT /
    pipeline-learned arms so ``stage_paper`` can still compile a paper
    whose rule-based numbers are real even when the neural components
    could not be trained in the current environment.
    """
    log.info("=== Stage: eval_rulebased_gold ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("Split: %d train / %d val / %d test",
             len(data.train), len(data.val), len(data.test))
    metrics = eval_rulebased_gold(config, data.test)
    log.info("Rule-based (gold OCR) F1=%.4f  per-field=%s",
             metrics.global_f1,
             {k: round(v, 4) for k, v in metrics.per_field_f1.items()})
    combined = combined_from_rulebased(config, metrics)
    combined.update(per_field_injection(metrics))
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)
