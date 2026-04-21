"""Three top-level stages: train, eval, paper."""
from __future__ import annotations

import json
import logging
import os
import statistics
import time
from pathlib import Path

from core.cost import summarise as cost_summarise
from core.errors import EvalError, TrainError
from core.seed import seed_everything
from core.telemetry import start_sampler, stop_sampler
from core.types import AssignerData, DataSplit, ExpConfig, Metrics, PipelineResult
from core.validate import validate_f1
from data.sroie import download_sroie, extract_crops, extract_receipt_regions, load_or_create_split
from models.assigner_train import train_assigner
from models.donut_eval import eval_donut
from models.donut_train import train_donut
from models.pipeline_eval import eval_pipeline
from models.rule_eval import (
    combined_from_rulebased,
    eval_rulebased_gold,
    per_field_injection,
)
from models.trocr_train import train_trocr
from models.yolo_train import train_yolo
from report.inject import expand_inputs, inject_results
from report.pdflatex import compile_paper_pdf

log = logging.getLogger("kaggle2")


# ---------------------------------------------------------------------------
# Telemetry helpers — failures must NEVER propagate to the caller.
# ---------------------------------------------------------------------------

_VASTAI_RATE_USD_HR: float = 0.50  # $/hr default; override via env if needed


def _start_telem(config: ExpConfig, stage: str) -> tuple[object, object, float]:
    """Start a telemetry sampler for *stage*; return (thread, event, t0).

    Any exception is caught and logged as a warning so that a missing
    ``nvidia-smi`` or permission error never aborts training.
    """
    out_path = os.path.join(config.output_dir, f"telemetry_{stage}.jsonl")
    try:
        thread, event = start_sampler(out_path, interval_s=5.0)
        return thread, event, time.monotonic()
    except Exception as exc:  # noqa: BLE001
        log.warning("Telemetry start failed (%s) — continuing without.", exc)
        return None, None, time.monotonic()


def _stop_telem(
    thread: object, event: object, t0: float, config: ExpConfig, stage: str
) -> None:
    """Stop the telemetry sampler and write a cost JSON summary."""
    elapsed = time.monotonic() - t0
    log.info("Stage '%s' wall-clock: %.1f s (%.2f min)", stage, elapsed, elapsed / 60)
    if thread is None or event is None:
        return
    try:
        from threading import Event as _Event
        from threading import Thread as _Thread

        assert isinstance(thread, _Thread) and isinstance(event, _Event)
        out_path = stop_sampler(thread, event)
        rate = float(os.environ.get("VASTAI_RATE_USD_HR", _VASTAI_RATE_USD_HR))
        cost = cost_summarise(out_path, rate)
        cost_path = os.path.join(config.output_dir, f"cost_{stage}.json")
        with open(cost_path, "w") as fh:
            json.dump(cost, fh, indent=2)
        log.info("Cost summary → %s", cost_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Telemetry stop/cost failed (%s) — continuing.", exc)


def _write_pipeline_meta(config: ExpConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump({"yolo_img_size": config.yolo_img_size}, f)


def _warn_below_expected(metrics: Metrics, config: ExpConfig, arch: str) -> None:
    """Soft-warn when F1 is below ``config.expected_f1_warn`` (not an error)."""
    if metrics.global_f1 < config.expected_f1_warn:
        log.warning(
            "%s F1=%.4f below expected_f1_warn=%.2f (not an error).",
            arch, metrics.global_f1, config.expected_f1_warn,
        )


def stage_train(config: ExpConfig) -> None:
    log.info("=== Stage: train ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("Split: %d train / %d val / %d test",
             len(data.train), len(data.val), len(data.test))
    if config.skip_donut:
        log.info("skip_donut=True — Phase 1 mode: DONUT training suppressed. "
                 "KD losses will be disabled downstream (kd_*_weight must be 0).")
        if config.kd_attn_weight != 0.0 or config.kd_logits_weight != 0.0:
            raise TrainError(
                "skip_donut=True but kd_attn_weight or kd_logits_weight != 0. "
                "Phase 1 cannot distil from a missing teacher; set both to 0.",
            )
    else:
        th, ev, t0 = _start_telem(config, "donut")
        try:
            donut_path = train_donut(config, data)
        finally:
            _stop_telem(th, ev, t0, config, "donut")
        log.info("DONUT → %s", donut_path)
    th_y, ev_y, t0_y = _start_telem(config, "yolo")
    try:
        yolo_path = train_yolo(config, data)
    finally:
        _stop_telem(th_y, ev_y, t0_y, config, "yolo")
    log.info("YOLO  → %s", yolo_path)
    crops = extract_crops(data.train, config.fields)
    regions = extract_receipt_regions(data.train, config.fields)
    log.info("Extracted %d labeled crops / %d receipt region-groups",
             len(crops), len(regions))
    if not crops:
        raise TrainError("No labeled SROIE crops — check box/ annotations.")
    th_t, ev_t, t0_t = _start_telem(config, "trocr")
    try:
        trocr_path = train_trocr(config, crops)
    finally:
        _stop_telem(th_t, ev_t, t0_t, config, "trocr")
    log.info("TrOCR → %s", trocr_path)
    assigner_data = AssignerData(trocr_path=trocr_path, crops=crops, regions=regions)
    assigner_path = train_assigner(config, assigner_data)
    log.info("Assigner → %s", assigner_path)
    _write_pipeline_meta(config)


def _eval_donut_or_skip(config: ExpConfig, data: DataSplit) -> Metrics:
    """Eval DONUT iff skip_donut is False AND a checkpoint exists on disk."""
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
    _warn_below_expected(dm, config, "donut")
    return dm


def _combined_metrics(
    config: ExpConfig, dm: Metrics, pm: PipelineResult, rb_gold: Metrics,
) -> dict[str, object]:
    out: dict[str, object] = {
        "donut_f1": dm.global_f1, "donut_ned": dm.global_ned, "donut_em": dm.global_em,
        "pipeline_f1": pm.assigner.global_f1,
        "pipeline_ned": pm.assigner.global_ned,
        "pipeline_em": pm.assigner.global_em,
        "rulebased_f1": pm.rulebased.global_f1,
        "rulebased_ned": pm.rulebased.global_ned,
        "rulebased_gold_f1": rb_gold.global_f1,
        "rulebased_gold_ned": rb_gold.global_ned,
        "f1_gap": round(dm.global_f1 - pm.assigner.global_f1, 4),
        "assigner_delta": round(pm.assigner.global_f1 - pm.rulebased.global_f1, 4),
        "donut_f1_company": dm.per_field_f1.get("company", 0.0),
        "donut_f1_date": dm.per_field_f1.get("date", 0.0),
        "donut_f1_address": dm.per_field_f1.get("address", 0.0),
        "donut_f1_total": dm.per_field_f1.get("total", 0.0),
        "rulebased_f1_company": rb_gold.per_field_f1.get("company", 0.0),
        "rulebased_f1_date": rb_gold.per_field_f1.get("date", 0.0),
        "rulebased_f1_address": rb_gold.per_field_f1.get("address", 0.0),
        "rulebased_f1_total": rb_gold.per_field_f1.get("total", 0.0),
        "epochs_donut": config.epochs_donut, "epochs_trocr": config.epochs_trocr,
        "epochs_yolo": config.epochs_yolo, "batch_size": config.batch_size,
        "lr": config.lr, "precision": config.precision,
        "label_smoothing": config.label_smoothing,
        "warmup_steps": config.warmup_steps,
        "yolo_img_size": config.yolo_img_size,
        "img_w": config.image_size[0], "img_h": config.image_size[1],
        "artifact_mode": "full",
    }
    return out


def stage_eval(config: ExpConfig, seeds: list[int] | None = None) -> None:
    """Run eval across one or more seeds; aggregate mean/std when ``seeds`` has
    more than one entry. Keeps the legacy single-seed keys for backwards
    compatibility with the paper's ``\\VAR{}`` substitution.
    """
    log.info("=== Stage: eval ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    run_seeds = seeds if seeds else [config.seed]
    donut_f1s: list[float] = []
    pipeline_f1s: list[float] = []
    last: dict[str, object] = {}
    for seed in run_seeds:
        if len(run_seeds) > 1:
            log.info("--- Eval seed=%d ---", seed)
        seed_everything(seed)
        dm = _eval_donut_or_skip(config, data)
        log.info("DONUT F1=%.4f", dm.global_f1)
        pm = eval_pipeline(config, data.test)
        validate_f1(pm.assigner, "pipeline")
        _warn_below_expected(pm.assigner, config, "pipeline")
        log.info("Pipeline (assigner)  F1=%.4f", pm.assigner.global_f1)
        log.info("Pipeline (rulebased) F1=%.4f", pm.rulebased.global_f1)
        # Also run rule-based on gold OCR so the paper's "Rule-based (gold OCR)"
        # row has a real number in full-pipeline mode too — this is a legitimate
        # ablation (assignment heuristic quality in isolation from OCR noise).
        rb_gold = eval_rulebased_gold(config, data.test)
        log.info("Rule-based (gold OCR) F1=%.4f", rb_gold.global_f1)
        donut_f1s.append(dm.global_f1)
        pipeline_f1s.append(pm.assigner.global_f1)
        last = _combined_metrics(config, dm, pm, rb_gold)
    if len(run_seeds) >= 2:
        last["donut_f1_mean"] = round(statistics.fmean(donut_f1s), 4)
        last["donut_f1_std"] = round(statistics.stdev(donut_f1s), 4)
        last["pipeline_f1_mean"] = round(statistics.fmean(pipeline_f1s), 4)
        last["pipeline_f1_std"] = round(statistics.stdev(pipeline_f1s), 4)
    last["seeds_used"] = list(run_seeds)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "combined_metrics.json"), "w") as f:
        json.dump(last, f, indent=2)


def stage_eval_rulebased_gold(config: ExpConfig) -> None:
    """Produce real rule-based F1 over SROIE gold-OCR (no HF/GPU dependency).

    Writes ``results/rulebased_gold_metrics.json`` and a ``combined_metrics.json``
    pre-populated with zeros for DONUT / pipeline-learned so ``stage_paper``
    can compile a paper whose rule-based numbers are honest real values even
    when the neural components could not be trained in this environment.
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


def stage_paper(config: ExpConfig) -> None:
    log.info("=== Stage: paper ===")
    metrics_path = os.path.join(config.output_dir, "combined_metrics.json")
    if not Path(metrics_path).exists():
        raise EvalError(f"Run eval stage first — {metrics_path} not found.")
    with open(metrics_path) as f:
        metrics: dict[str, object] = json.load(f)
    # Best-effort: merge telemetry/cost JSON files into metrics dict.
    try:
        from report.figures import render_all as _render_all
        _render_all(config.output_dir)
    except Exception as exc:  # noqa: BLE001
        log.warning("figures.render_all failed (%s) — continuing.", exc)
    for stage in ("donut", "yolo", "trocr", "pipeline"):
        cost_path = os.path.join(config.output_dir, f"cost_{stage}.json")
        if Path(cost_path).exists():
            try:
                with open(cost_path) as fh:
                    cost_data: dict[str, object] = json.load(fh)
                for k, v in cost_data.items():
                    metrics.setdefault(f"{stage}_{k}", v)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to merge cost_%s.json: %s", stage, exc)
    with open(config.paper_template) as f:
        template = f.read()
    # Inline \input{sections/...} before \VAR{} substitution — keeps the
    # 166-LOC rule applicable to each section file while producing a single
    # flat paper_filled.tex that tectonic can compile without extra paths.
    template = expand_inputs(template, Path(config.paper_template).parent)
    filled = inject_results(template, metrics)
    tex_out = Path(config.paper_output)
    tex_out.parent.mkdir(parents=True, exist_ok=True)
    with open(tex_out, "w") as f:
        f.write(filled)
    log.info("Paper LaTeX written to %s", tex_out)
    bib_src = Path(config.paper_template).parent / "references.bib"
    pdf = compile_paper_pdf(tex_out, bib_src)
    if pdf is not None:
        log.info("Paper PDF written to %s", pdf)
