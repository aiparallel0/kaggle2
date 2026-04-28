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


def _emit_foundation_metrics(config: ExpConfig, test: list) -> None:  # type: ignore[type-arg]
    """P4 — write ``foundation_metrics.json`` to ``config.output_dir``.

    Calls :func:`models.foundation_oracle.foundation_predict` for each
    test receipt (results cached by content hash) and reduces through
    the shared ``compute_metrics`` so the numbers are directly
    comparable with DONUT/pipeline F1/NED/EM.  Any missing API key
    yields an empty Receipt → metrics degrade gracefully to 0.0.
    """
    try:
        from core.metrics import compute_metrics
        from core.types import EvalBundle
        from models.foundation_oracle import foundation_predict
    except ImportError as exc:
        log.warning("foundation arm: %s — skipping side-car emit", exc)
        return
    preds = [foundation_predict(r.image_path, config) for r in test]
    bundle = EvalBundle(
        predictions=preds, receipts=test, fields=list(config.fields),
    )
    m = compute_metrics(bundle)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    out = Path(config.output_dir) / "foundation_metrics.json"
    out.write_text(json.dumps({
        "api": config.foundation_api,
        "foundation_f1": round(m.global_f1, 4),
        "foundation_ned": round(m.global_ned, 4),
        "foundation_em": round(m.global_em, 4),
        "per_field_f1": {k: round(v, 4) for k, v in m.per_field_f1.items()},
        "n_test": len(test),
    }, indent=2))
    log.info("foundation_metrics.json: F1=%.4f NED=%.4f EM=%.4f",
             m.global_f1, m.global_ned, m.global_em)


def _emit_focus_diagnostics(config: ExpConfig) -> None:
    """PR-FOCUS — record which FOCUS sub-heads are configured for this run.

    Writes ``runs/<run_id>/metrics/focus_diagnostics.json`` so the paper
    stage and downstream auditors can verify which factored decoder
    (FOCUS-A / FOCUS-T / FOCUS-C) was active without grepping the run
    config.  Best-effort: a write failure is logged but never blocks
    eval (mirrors :func:`write_env_snapshot`).
    """
    try:
        out_dir = Path(config.output_dir) / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "framework": "FOCUS — Field-specific Output heads with "
                         "Cohesive Unified Selection",
            "focus_enabled": bool(config.focus_enabled),
            "sub_heads": {
                "FOCUS-A": {  # span head, address — PR #106
                    "enabled": bool(config.focus_enabled),
                    "max_span": int(config.focus_max_span),
                    "iou_weight": float(config.focus_iou_weight),
                    "boundary_weight": float(config.focus_boundary_weight),
                },
                "FOCUS-T": {  # relational head, total
                    "enabled": bool(
                        config.focus_enabled and config.focus_total_enabled,
                    ),
                    "witness_weight": float(config.focus_total_witness_weight),
                },
                "FOCUS-C": {  # positional head, company
                    "enabled": bool(
                        config.focus_enabled and config.focus_company_enabled,
                    ),
                    "y_weight": float(config.focus_company_y_weight),
                    "boilerplate_weight": float(
                        config.focus_company_boilerplate_weight,
                    ),
                },
                "FOCUS-D": {  # date — stays on the existing point head
                    "enabled": False,
                    "note": "regex-conforming substring, near-saturated by "
                            "token-level cross-attention",
                },
            },
            "priors_v4": bool(config.priors_v4),
        }
        (out_dir / "focus_diagnostics.json").write_text(
            json.dumps(payload, indent=2),
        )
    except OSError as exc:  # pragma: no cover — diagnostics are best-effort
        log.warning("focus_diagnostics emit failed: %s", exc)


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


def _zero_metrics(fields: list[str]) -> Metrics:
    """A no-information Metrics placeholder (used when an arm is skipped)."""
    return Metrics(
        global_f1=0.0, global_ned=0.0, global_em=0.0,
        per_field_f1={f: 0.0 for f in fields},
        per_field_ned={f: 0.0 for f in fields},
        per_field_em={f: 0.0 for f in fields},
    )


# Keys that exist *only* because the GT-OCR-rulebased baseline ran.  When
# the canonical 347-image test set is in use the GT-OCR arm cannot run
# (no GT box files in the Task-3 archive), so these keys are stripped
# rather than emitted as zeros — the paper's basic vs advanced templates
# read different key sets and never reference the missing ones.
_GTOCR_STRIP_PREFIXES: tuple[str, ...] = (
    "gtocr_rulebased_", "rulebased_", "oracle_patch_",
)


def _strip_gtocr_keys(d: dict[str, object]) -> None:
    for k in [k for k in d if k.startswith(_GTOCR_STRIP_PREFIXES)]:
        d.pop(k, None)


def stage_eval(
    config: ExpConfig, seeds: list[int] | None = None,
    oracle_address: bool = False,
) -> None:
    """Run eval across one or more seeds; aggregate mean/std/CI.

    Keeps the legacy single-seed keys (``donut_f1``, ``pipeline_f1``,
    ``assigner_delta``) populated for back-compat with the paper's
    \\VAR{} substitution.  Multi-seed runs additionally emit
    ``*_mean`` / ``*_std`` / ``*_ci_lo`` / ``*_ci_hi`` so the paper can
    render seed-level uncertainty bands.  For n=1 the paper-side bootstrap
    CI over per-image correctness is still available via
    :func:`core.statistics.bootstrap_ci`.

    When ``oracle_address`` is True the harness short-circuits to the
    Day-1 FOCUS gate: it loads the SROIE split, runs Tier A (clean
    box-file ceiling) + Tier B (canonical-347 heuristic ceiling) via
    :func:`models.pipeline_oracle.compute_oracle_address`, writes
    ``runs/<run_id>/metrics/oracle_address.json``, logs the decision
    branch, and returns without touching any other sidecar.
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
    _emit_focus_diagnostics(config)
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    if oracle_address:
        from models.pipeline_oracle import compute_oracle_address
        payload = compute_oracle_address(data, config)
        out = Path(config.output_dir) / "metrics" / "oracle_address.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        log.info("Oracle address sidecar -> %s", out)
        log.info(
            "Tier B canonical: f1=%.4f em=%.4f decision=%s p1_passes=%s",
            payload["tier_b_canonical_heuristic"]["f1"],
            payload["tier_b_canonical_heuristic"]["em"],
            payload["decision"], payload["p1_passes"],
        )
        return
    run_seeds = list(seeds) if seeds else list(config.seeds[: config.n_trials])
    log.info("Eval harness: n_trials=%d seeds=%s", len(run_seeds), run_seeds)
    canonical = bool(getattr(config, "canonical_sroie_enabled", False))
    if canonical:
        log.info("Canonical SROIE 347-image test set active — skipping "
                 "GT-OCR-rulebased baseline and oracle-patch diagnostic "
                 "(no GT boxes available in the Task-3 archive).")
    donut_f1s: list[float] = []
    pipeline_f1s: list[float] = []
    gtocr_rulebased_f1s: list[float] = []
    # v4 — track per-field, NED, and EM across seeds so the inject
    # layer can render ``mean ± std`` for every headline cell, not
    # just the global F1.  Indexed by metric key (e.g. ``donut_ned``,
    # ``donut_f1_company``); values are the per-seed point estimates
    # collected across the loop below.
    extra_seed_metrics: dict[str, list[float]] = {}
    last: dict[str, object] = {}
    last_donut_preds: list[Prediction] = []
    last_pipeline_preds: list[Prediction] = []
    last_donut_metrics: Metrics | None = None
    last_pipeline_metrics: Metrics | None = None
    for seed in run_seeds:
        if len(run_seeds) > 1:
            log.info("--- Eval seed=%d ---", seed)
        seed_everything(seed)
        dm, dp = _eval_donut_or_skip(config, data)
        log.info("DONUT F1=%.4f", dm.global_f1)
        pm = eval_pipeline(config, data.test)
        validate_f1(pm.assigner, "pipeline",
                    os.path.join(config.output_dir, "pipeline_metrics.json"))
        warn_below_expected(pm.assigner, config, "pipeline")
        warn_pipeline_diagnostics(config)
        log.info("Pipeline (hybrid)              F1=%.4f", pm.assigner.global_f1)
        log.info("Pipeline (TrOCR-regex)         F1=%.4f", pm.rulebased.global_f1)
        if canonical:
            gtocr_rb = _zero_metrics(list(config.fields))
            patched_assigner = pm.assigner
        else:
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
        if not canonical:
            gtocr_rulebased_f1s.append(gtocr_rb.global_f1)
        # v4 — record per-field F1 + NED + EM for every seed so the
        # downstream aggregator can emit mean ± std for each cell.
        # Keyed identically to the headline metric keys (so
        # ``\VAR{donut_f1_company:mean_std_pct1}`` resolves directly).
        per_seed_arms: tuple[tuple[str, Metrics], ...] = (
            ("donut", dm), ("pipeline", pm.assigner),
        )
        if not canonical:
            per_seed_arms = per_seed_arms + (("rulebased", gtocr_rb),)
        for sysname, m in per_seed_arms:
            extra_seed_metrics.setdefault(f"{sysname}_ned", []).append(m.global_ned)
            extra_seed_metrics.setdefault(f"{sysname}_em", []).append(m.global_em)
            for fld in ("company", "date", "address", "total"):
                extra_seed_metrics.setdefault(
                    f"{sysname}_f1_{fld}", [],
                ).append(m.per_field_f1.get(fld, 0.0))
        last = build_combined(config, dm, pm, gtocr_rb)
        if canonical:
            _strip_gtocr_keys(last)
        else:
            last["oracle_patch_f1_if_applied"] = round(patched_assigner.global_f1, 4)
        last_donut_preds = dp
        last_pipeline_preds = pm.assigner_preds
        last_donut_metrics = dm
        last_pipeline_metrics = pm.assigner
    # Always emit the variance block, even when n=1 — downstream consumers
    # (paper \VAR{}, log dashboards) should not branch on seed count.
    _aggregate_seed_variance(donut_f1s, "donut_f1", last)
    _aggregate_seed_variance(pipeline_f1s, "pipeline_f1", last)
    if not canonical:
        _aggregate_seed_variance(gtocr_rulebased_f1s, "gtocr_rulebased_f1", last)
    # v4 — aggregate the per-field / NED / EM seed series the same way.
    # Note ``rulebased_*`` keys map to the ``gtocr_rulebased_*`` namespace
    # the rest of the paper uses; rename on emit to keep the inject keys
    # consistent.
    for key, seq in extra_seed_metrics.items():
        out_key = key.replace("rulebased_", "gtocr_rulebased_")
        if canonical and out_key.startswith(_GTOCR_STRIP_PREFIXES):
            continue
        _aggregate_seed_variance(seq, out_key, last)
    # Per-image bootstrap CI + paired McNemar: uses the last run's
    # per-image vectors populated by ``compute_metrics`` via
    # ``build_combined``.  The McNemar test runs on the binary
    # all-fields-EM vector (``per_image_correct``) — that's the right
    # signal for "did this image change between systems".  The
    # bootstrap CI on the headline F1 metric uses the per-image
    # macro-F1 vector (``per_image_f1``); the all-fields-EM vector is
    # degenerate (every entry zero) whenever no receipt has every
    # field correct simultaneously, which produced the zero-width
    # ``pipeline_bootstrap_ci_*`` keys in earlier runs.  The
    # all-fields-EM-based CI is still emitted under ``*_em_*`` keys so
    # the paper can quote both quantities without ambiguity.  Paired
    # tests are valid because DONUT and pipeline are evaluated on the
    # same 63 test images in fixed order.
    d_raw = last.get("donut_per_image_correct")
    p_raw = last.get("pipeline_per_image_correct")
    d_vec = [bool(x) for x in d_raw] if isinstance(d_raw, list) else []
    p_vec = [bool(x) for x in p_raw] if isinstance(p_raw, list) else []
    d_f1_raw = last.get("donut_per_image_f1")
    p_f1_raw = last.get("pipeline_per_image_f1")
    d_f1_vec = [float(x) for x in d_f1_raw] if isinstance(d_f1_raw, list) else []
    p_f1_vec = [float(x) for x in p_f1_raw] if isinstance(p_f1_raw, list) else []
    if p_f1_vec:
        lo, hi = bootstrap_ci(
            p_f1_vec,
            n_iter=config.bootstrap_n_iter,
            ci_level=config.bootstrap_ci_level,
        )
        last["pipeline_bootstrap_ci_lo"] = round(lo, 4)
        last["pipeline_bootstrap_ci_hi"] = round(hi, 4)
    if p_vec:
        em_lo, em_hi = bootstrap_ci(
            p_vec,
            n_iter=config.bootstrap_n_iter,
            ci_level=config.bootstrap_ci_level,
        )
        last["pipeline_em_bootstrap_ci_lo"] = round(em_lo, 4)
        last["pipeline_em_bootstrap_ci_hi"] = round(em_hi, 4)
    if d_f1_vec and p_f1_vec and len(d_f1_vec) == len(p_f1_vec):
        _, ci_lo, ci_hi = paired_bootstrap_delta_ci(
            d_f1_vec, p_f1_vec,
            n_iter=config.bootstrap_n_iter,
            ci_level=config.bootstrap_ci_level,
        )
        last["delta_f1_ci_lo"] = round(ci_lo, 4)
        last["delta_f1_ci_hi"] = round(ci_hi, 4)
    if d_vec and p_vec and len(d_vec) == len(p_vec):
        _, em_ci_lo, em_ci_hi = paired_bootstrap_delta_ci(
            d_vec, p_vec,
            n_iter=config.bootstrap_n_iter,
            ci_level=config.bootstrap_ci_level,
        )
        last["delta_em_ci_lo"] = round(em_ci_lo, 4)
        last["delta_em_ci_hi"] = round(em_ci_hi, 4)
        # Full-precision float: ``report.inject._format_pvalue`` handles
        # rendering (``p=3e-5`` → ``$3.0\times 10^{-5}$`` not ``0.0000``).
        last["mcnemar_p"] = float(mcnemar(d_vec, p_vec))
    last["seeds_used"] = list(run_seeds)
    last["n_trials"] = len(run_seeds)
    last["bootstrap_n_iter"] = config.bootstrap_n_iter
    last["bootstrap_ci_level"] = config.bootstrap_ci_level
    # Surface the eval split in combined_metrics so the paper template
    # (basic vs advanced variant) can branch on it without parsing
    # config.json again.  ``test_set_kind`` ∈ {"canonical_347", "internal_63"}.
    last["test_set_kind"] = "canonical_347" if canonical else "internal_63"
    last["test_set_size"] = 347 if canonical else 63
    if canonical:
        # Final defensive strip — any merge above must not reintroduce
        # GT-OCR-rulebased / oracle-patch keys when canonical is active.
        _strip_gtocr_keys(last)
    # P4 — foundation-model ceiling arm (Claude Sonnet / GPT-4V zero-shot).
    # Opt-in via ``config.foundation_enabled``; API keys + caching are
    # owned by :mod:`models.foundation_oracle`.  Written as a side-car
    # so it never alters the headline pipeline/DONUT metrics.
    if getattr(config, "foundation_enabled", False):
        _emit_foundation_metrics(config, data.test)
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
