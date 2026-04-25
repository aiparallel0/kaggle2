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
    config: ExpConfig, dm: Metrics, pm: PipelineResult, gtocr_rb: Metrics,
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
        "gtocr_rulebased_f1": gtocr_rb.global_f1,
        "gtocr_rulebased_ned": gtocr_rb.global_ned,
        # GT-OCR-stream exact-match rate — surfaced so Table I's baseline
        # EM cell resolves to a concrete number instead of the ``---``
        # backstop (source of truth ``gtocr_rulebased_metrics.json`` →
        # ``global_em``).
        "gtocr_rulebased_em": gtocr_rb.global_em,
        "f1_gap": round(dm.global_f1 - pm.assigner.global_f1, 4),
        "assigner_delta": round(pm.assigner.global_f1 - pm.rulebased.global_f1, 4),
        "donut_f1_company": dm.per_field_f1.get("company", 0.0),
        "donut_f1_date": dm.per_field_f1.get("date", 0.0),
        "donut_f1_address": dm.per_field_f1.get("address", 0.0),
        "donut_f1_total": dm.per_field_f1.get("total", 0.0),
        # Pipeline per-field F1 — Table IV row + Fig. 9 panel (S4, S8).
        "pipeline_f1_company": pm.assigner.per_field_f1.get("company", 0.0),
        "pipeline_f1_date": pm.assigner.per_field_f1.get("date", 0.0),
        "pipeline_f1_address": pm.assigner.per_field_f1.get("address", 0.0),
        "pipeline_f1_total": pm.assigner.per_field_f1.get("total", 0.0),
        "rulebased_f1_company": gtocr_rb.per_field_f1.get("company", 0.0),
        "rulebased_f1_date": gtocr_rb.per_field_f1.get("date", 0.0),
        "rulebased_f1_address": gtocr_rb.per_field_f1.get("address", 0.0),
        "rulebased_f1_total": gtocr_rb.per_field_f1.get("total", 0.0),
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
        # Seed harness provenance — surfaced so the paper's Reproducibility
        # subsection and the \VAR{n_trials} / \VAR{seeds_str} placeholders
        # resolve without branching on dict presence.
        "n_trials": config.n_trials,
        "seeds_configured": list(config.seeds),
        "bootstrap_n_iter": config.bootstrap_n_iter,
        "bootstrap_ci_level": config.bootstrap_ci_level,
        # Assigner architecture knobs — exposed so §IV.C's
        # parameter-count description and the d / L numbers can be
        # rendered by the inject layer rather than hard-coded literals
        # (which historically drifted from the shipped checkpoint as
        # the assigner was scaled from the 161K-parameter d=128, L=2
        # variant to the 1.16M-parameter d=384, L=6 variant).
        "assigner_hidden": config.assigner_hidden,
        "assigner_n_layers_level2": config.assigner_n_layers_level2,
        # Per-image correctness vectors (all-fields EM per receipt):
        # consumed by stages/eval.py for the McNemar test and the
        # ``*_em_*`` paired-bootstrap CIs.
        "donut_per_image_correct": list(dm.per_image_correct),
        "pipeline_per_image_correct": list(pm.assigner.per_image_correct),
        # Per-image macro-F1 vectors (mean of per-field token-F1):
        # consumed by stages/eval.py for the headline ``pipeline_bootstrap_ci_*``
        # and ``delta_f1_ci_*`` keys.  Computed per receipt so the CI
        # tracks the headline F1 metric directly — the all-fields-EM
        # vector is degenerate whenever no receipt has every field
        # correct simultaneously, which produced the zero-width
        # ``pipeline_bootstrap_ci_*`` keys in earlier runs.
        "donut_per_image_f1": list(dm.per_image_f1),
        "pipeline_per_image_f1": list(pm.assigner.per_image_f1),
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
            for key in ("empty_detection_fraction", "per_receipt_error_fraction",
                        "donut_params_m", "pipeline_params_m"):
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
    """Fold every ``cost_<stage>.json`` into the metrics dict in place.

    When ``cost_pipeline.json`` is absent (the common case — we only
    measure per-stage cost during ``--stage train``), synthesise the
    headline ``pipeline_<key>`` aggregates from the per-stage cost
    files so Table II's pipeline column resolves: cost / energy / CO₂
    / minutes are summed over YOLO + TrOCR + assigner; peak VRAM
    takes the max (the binding constraint at any single moment).
    """
    sums: dict[str, float] = {}
    peak_vrams: list[float] = []
    have_pipeline_file = Path(
        os.path.join(config.output_dir, "cost_pipeline.json")).exists()
    for stage in ("donut", "yolo", "trocr", "assigner", "pipeline"):
        cost_path = os.path.join(config.output_dir, f"cost_{stage}.json")
        if not Path(cost_path).exists():
            continue
        try:
            with open(cost_path) as fh:
                cost_data: dict[str, object] = json.load(fh)
            for k, v in cost_data.items():
                metrics.setdefault(f"{stage}_{k}", v)
            if not have_pipeline_file and stage in ("yolo", "trocr", "assigner"):
                for k in ("cost_usd", "energy_kwh", "co2_kg", "train_minutes"):
                    val = cost_data.get(k)
                    if isinstance(val, int | float):
                        sums[k] = sums.get(k, 0.0) + float(val)
                pv = cost_data.get("peak_vram_gb")
                if isinstance(pv, int | float):
                    peak_vrams.append(float(pv))
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to merge cost_%s.json: %s", stage, exc)
    if not have_pipeline_file:
        for k, v in sums.items():
            metrics.setdefault(f"pipeline_{k}", round(v, 4))
        if peak_vrams:
            metrics.setdefault("pipeline_peak_vram_gb", round(max(peak_vrams), 4))
