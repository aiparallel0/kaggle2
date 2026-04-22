"""Build and enrich the ``combined_metrics.json`` blob driving \\VAR{}.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: single source of truth for the flat ``dict[str, object]`` that
    ``report.inject.inject_results`` consumes.  Houses both the
    ``build_combined`` assembler used by the eval stage and the
    merge helpers used by the paper stage so every ``\\VAR{}`` key
    surfaced in the LaTeX paper has a provenance trail to a JSON file
    on disk (``combined_metrics.json``, ``assigner_metrics.json``,
    ``pipeline_metrics.json``, ``pipeline_meta.json``, ``cost_*.json``).
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from core.types import ExpConfig, Metrics, PipelineResult

log = logging.getLogger("kaggle2")


def build_combined(
    config: ExpConfig, dm: Metrics, pm: PipelineResult, rb_gold: Metrics,
) -> dict[str, object]:
    """Assemble the base ``combined_metrics.json`` dict for one eval run.

    Emits every field the paper's \\VAR{} substitution depends on,
    including the differential-LR pair, KD hook weights (reported as
    zero so reviewers can see they were inactive), and the
    ``epochs_assigner`` knob.  Richer keys (parameter counts,
    diagnostic fractions, parity flag) are merged in by
    ``stage_paper`` via :func:`merge_assigner_metrics` and
    :func:`merge_pipeline_diagnostics`.
    """
    return {
        "donut_f1": dm.global_f1, "donut_ned": dm.global_ned, "donut_em": dm.global_em,
        "pipeline_f1": pm.assigner.global_f1,
        "pipeline_ned": pm.assigner.global_ned,
        "pipeline_em": pm.assigner.global_em,
        "rulebased_f1": pm.rulebased.global_f1,
        "rulebased_ned": pm.rulebased.global_ned,
        "rulebased_gold_f1": rb_gold.global_f1,
        "rulebased_gold_ned": rb_gold.global_ned,
        # Rule-based exact-match rate — surfaced so Table I's rule-based
        # EM cell resolves to a concrete number instead of the ``---``
        # backstop (paper_corrections.md item 6; source of truth
        # ``rulebased_gold_metrics.json`` → ``global_em``).
        "rulebased_gold_em": rb_gold.global_em,
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
        "epochs_yolo": config.epochs_yolo, "epochs_assigner": config.epochs_assigner,
        "batch_size": config.batch_size,
        "lr": config.lr, "precision": config.precision,
        "label_smoothing": config.label_smoothing,
        "warmup_steps": config.warmup_steps,
        "yolo_img_size": config.yolo_img_size,
        "img_w": config.image_size[0], "img_h": config.image_size[1],
        "artifact_mode": "full",
        # Differential LR: the DONUT decoder is trained 10× faster than
        # the encoder to fit the resized embedding rows for the
        # SROIE-specific special tokens (<s_sroie>, <s_company>, ...).
        "lr_encoder": config.lr,
        "lr_decoder": config.lr_decoder,
        # KD scaffolding: currently off — reported explicitly so
        # reviewers can see they did not influence results.
        "kd_attn_weight": config.kd_attn_weight,
        "kd_logits_weight": config.kd_logits_weight,
    }


def merge_assigner_metrics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Surface ``assigner_metrics.json`` keys for \\VAR{} substitution."""
    path = os.path.join(config.output_dir, "assigner_metrics.json")
    if not Path(path).exists():
        return
    try:
        with open(path) as fh:
            am: dict[str, object] = json.load(fh)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed to merge assigner_metrics.json: %s", exc)
        return
    n_params = am.get("n_params")
    if isinstance(n_params, int | float):
        # Store as int so the paper renders ``400\,K`` rather than
        # ``400.0000\,K`` (paper_corrections.md item 1 — the default
        # float formatter in ``report.inject`` uses ``{:.4f}``).
        metrics["assigner_params_k"] = int(round(float(n_params) / 1000.0))
    for src, dst in (
        ("best_epoch", "assigner_best_epoch"),
        ("stopped_at_epoch", "assigner_stopped_at"),
        ("best_val_loss", "assigner_best_val_loss"),
    ):
        if am.get(src) is not None:
            metrics[dst] = am[src]


def merge_pipeline_diagnostics(
    config: ExpConfig, metrics: dict[str, object],
) -> None:
    """Surface pipeline per-receipt diagnostic fractions + parity flag."""
    path = os.path.join(config.output_dir, "pipeline_metrics.json")
    if Path(path).exists():
        try:
            with open(path) as fh:
                pm: dict[str, object] = json.load(fh)
            for key in ("empty_detection_fraction", "per_receipt_error_fraction"):
                if key in pm:
                    metrics[key] = pm[key]
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to merge pipeline_metrics.json: %s", exc)
    # Parity flag: pipeline_meta.json must report the same yolo_img_size
    # that the live config carries, or Bug 5 silently re-enters.
    meta = Path(config.output_dir) / "pipeline_meta.json"
    if meta.exists():
        try:
            with open(meta) as fh:
                meta_data = json.load(fh)
            metrics["parity_ok"] = (
                int(meta_data.get("yolo_img_size", -1)) == config.yolo_img_size
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to read pipeline_meta.json: %s", exc)


def merge_cost_json(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold every ``cost_<stage>.json`` into the metrics dict in place."""
    for stage in ("donut", "yolo", "trocr", "pipeline"):
        cost_path = os.path.join(config.output_dir, f"cost_{stage}.json")
        if not Path(cost_path).exists():
            continue
        try:
            with open(cost_path) as fh:
                cost_data: dict[str, object] = json.load(fh)
            for k, v in cost_data.items():
                metrics.setdefault(f"{stage}_{k}", v)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to merge cost_%s.json: %s", stage, exc)
