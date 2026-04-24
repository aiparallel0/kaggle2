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

from core.env_snapshot import write_env_snapshot
from core.errors import EvalError
from core.seed import seed_everything
from core.statistics import bootstrap_ci, mcnemar, paired_bootstrap_delta_ci
from core.types import DataSplit, ExpConfig, Metrics, Prediction
from core.validate import validate_f1
from data.sroie import download_sroie, load_or_create_split
from models.donut_eval import eval_donut
from models.pipeline_eval import eval_pipeline
from models.rule_eval import (
    combined_from_rulebased,
    eval_gtocr_rulebased,
    per_field_injection,
)
from report.combine import build_combined
from stages._common import (
    assert_hybrid_beats_gtocr_rulebased,
    oracle_patch_hybrid,
    warn_below_expected,
    warn_pipeline_diagnostics,
)
from stages.eval_producers import emit_all

log = logging.getLogger("kaggle2")


def _eval_donut_or_skip(
    config: ExpConfig, data: DataSplit,
) -> tuple[Metrics, list[Prediction]]:
    """Eval DONUT iff ``skip_donut`` is False AND a checkpoint exists."""
    donut_model = os.path.join(config.output_dir, "donut")
    if config.skip_donut:
        log.info("skip_donut=True — skipping DONUT eval; reporting zeros.")
        zeros = Metrics(
            global_f1=0.0, global_ned=0.0, global_em=0.0,
            per_field_f1={f: 0.0 for f in config.fields},
            per_field_ned={f: 0.0 for f in config.fields},
            per_field_em={f: 0.0 for f in config.fields},
        )
        return zeros, []
    if not Path(donut_model).exists():
        raise EvalError(
            f"DONUT checkpoint not found at {donut_model}. Either run train "
            "stage first, or set skip_donut=true for a Phase-1 pipeline-only run.",
        )
    dm, dp = eval_donut(config, data.test)
    validate_f1(dm, "donut")
    warn_below_expected(dm, config, "donut")
    return dm, dp


def _per_seed_metrics(
    config: ExpConfig, data: DataSplit, seed: int,
) -> tuple[Metrics, Metrics, Metrics]:
    """Run one (DONUT, pipeline, GT-OCR-stream-rulebased) eval triple at `seed`."""
    seed_everything(seed)
    dm, _ = _eval_donut_or_skip(config, data)
    log.info("DONUT F1=%.4f", dm.global_f1)
    pm = eval_pipeline(config, data.test)
    validate_f1(pm.assigner, "pipeline",
                os.path.join(config.output_dir, "pipeline_metrics.json"))
    warn_below_expected(pm.assigner, config, "pipeline")
    warn_pipeline_diagnostics(config)
    log.info("Pipeline (hybrid)              F1=%.4f", pm.assigner.global_f1)
    log.info("Pipeline (TrOCR-regex)         F1=%.4f", pm.rulebased.global_f1)
    gtocr_rb = eval_gtocr_rulebased(config, data.test)
    log.info("Baseline (GT-OCR-stream regex) F1=%.4f", gtocr_rb.global_f1)
    assert_hybrid_beats_gtocr_rulebased(pm.assigner, gtocr_rb)
    # Change F (diagnostic-only): oracle_patch_hybrid writes
    # ``oracle_patched_fields.json`` so reviewers can see how much
    # headroom rule-based patching *would* provide, but the returned
    # post-patch metrics are NOT substituted into the headline hybrid
    # F1.  See follow-up Fix A — the previous "pm.assigner = patched"
    # clobbered a true 0.7993 hybrid run with the 0.5824 post-patch
    # number in ``combined_metrics.pipeline_f1``.
    oracle_patch_hybrid(pm, gtocr_rb, config, data.test)
    return dm, pm.assigner, gtocr_rb


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
    # Best-effort env snapshot; never block eval on a missing config.json.
    try:
        env_dir = Path(config.output_dir) / "env"
        cfg_path = Path(os.environ.get("KAGGLE2_CONFIG_PATH", "config.json"))
        write_env_snapshot(
            env_dir, cfg_path,
            run_id=Path(config.output_dir).name,
            seed=int(config.seeds[0]) if config.seeds else 0,
        )
    except OSError as exc:  # pragma: no cover — env snapshot is best-effort
        log.warning("env_snapshot failed: %s", exc)
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    run_seeds = list(seeds) if seeds else list(config.seeds[: config.n_trials])
    log.info("Eval harness: n_trials=%d seeds=%s", len(run_seeds), run_seeds)
    donut_f1s: list[float] = []
    pipeline_f1s: list[float] = []
    gtocr_rulebased_f1s: list[float] = []
    last: dict[str, object] = {}
    last_donut_preds: list[Prediction] = []
    last_pipeline_preds: list[Prediction] = []
    last_donut_metrics: Metrics | None = None
    last_pipeline_metrics: Metrics | None = None
    for seed in run_seeds:
        if len(run_seeds) > 1:
            log.info("--- Eval seed=%d ---", seed)
        dm, dp = _eval_donut_or_skip(config, data)
        log.info("DONUT F1=%.4f", dm.global_f1)
        seed_everything(seed)
        pm = eval_pipeline(config, data.test)
        validate_f1(pm.assigner, "pipeline",
                    os.path.join(config.output_dir, "pipeline_metrics.json"))
        warn_below_expected(pm.assigner, config, "pipeline")
        warn_pipeline_diagnostics(config)
        log.info("Pipeline (hybrid)              F1=%.4f", pm.assigner.global_f1)
        log.info("Pipeline (TrOCR-regex)         F1=%.4f", pm.rulebased.global_f1)
        gtocr_rb = eval_gtocr_rulebased(config, data.test)
        log.info("Baseline (GT-OCR-stream regex) F1=%.4f", gtocr_rb.global_f1)
        assert_hybrid_beats_gtocr_rulebased(pm.assigner, gtocr_rb)
        # Fix A (follow-up): oracle_patch_hybrid is DIAGNOSTIC-ONLY.
        # Its post-patch F1 is surfaced as ``oracle_patch_f1_if_applied``
        # further down; the headline ``pipeline_f1`` key stays bound to
        # the real hybrid ``pm.assigner.global_f1``.
        patched_assigner = oracle_patch_hybrid(pm, gtocr_rb, config, data.test)
        donut_f1s.append(dm.global_f1)
        pipeline_f1s.append(pm.assigner.global_f1)
        gtocr_rulebased_f1s.append(gtocr_rb.global_f1)
        last = build_combined(config, dm, pm, gtocr_rb)
        last["oracle_patch_f1_if_applied"] = round(patched_assigner.global_f1, 4)
        last_donut_preds = dp
        last_pipeline_preds = pm.assigner_preds
        last_donut_metrics = dm
        last_pipeline_metrics = pm.assigner
    # Always emit the variance block, even when n=1 — downstream consumers
    # (paper \VAR{}, log dashboards) should not branch on seed count.
    _aggregate_seed_variance(donut_f1s, "donut_f1", last)
    _aggregate_seed_variance(pipeline_f1s, "pipeline_f1", last)
    _aggregate_seed_variance(gtocr_rulebased_f1s, "gtocr_rulebased_f1", last)
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
    # Producer: write per-sample predictions, per_field_errors.jsonl, and
    # extended_metrics.json from the real eval data of the last seed so
    # every new \VAR{} resolves to a measured value in the PDF.
    emit_all(
        config.output_dir, tuple(config.fields),
        donut_preds=last_donut_preds or None,
        pipeline_preds=last_pipeline_preds or None,
        receipts=data.test,
        donut_metrics=last_donut_metrics,
        pipeline_metrics=last_pipeline_metrics,
        n_iter=config.bootstrap_n_iter,
        level=config.bootstrap_ci_level,
    )


def stage_eval_gtocr_rulebased(config: ExpConfig) -> None:
    """GT-OCR-stream rule-based F1 — no HF / GPU dependency.

    Bypasses YOLO+TrOCR by feeding SROIE ground-truth box text/bboxes
    directly into ``rule_based_assign`` (the same function the live
    pipeline uses).  Writes ``results/gtocr_rulebased_metrics.json`` and a
    ``combined_metrics.json`` pre-populated with zeros for the DONUT /
    pipeline-learned arms so ``stage_paper`` can still compile a paper
    whose rule-based numbers are real even when the neural components
    could not be trained in the current environment.
    """
    log.info("=== Stage: eval_gtocr_rulebased ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("Split: %d train / %d val / %d test",
             len(data.train), len(data.val), len(data.test))
    metrics = eval_gtocr_rulebased(config, data.test)
    log.info("Baseline (GT-OCR-stream regex) F1=%.4f  per-field=%s",
             metrics.global_f1,
             {k: round(v, 4) for k, v in metrics.per_field_f1.items()})
    combined = combined_from_rulebased(config, metrics)
    combined.update(per_field_injection(metrics))
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(combined, f, indent=2)
