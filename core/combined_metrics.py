"""Typed view of ``results/combined_metrics.json`` — the paper's data contract.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: declarative schema enumerating every ``\\VAR{}`` placeholder that
    the IEEE paper template may consume. The TypedDict has
    ``total=False`` because keys are populated incrementally across
    the eval, paper, and multi-seed aggregation stages: a single-seed
    CPU-only run still yields a well-typed (if sparse)
    :class:`CombinedMetrics`. Separating the schema from the reducers
    in :mod:`core.metrics` keeps both modules under the 166-LOC cap
    and gives authors a single place to consult when a new \\VAR{}
    key is added to the paper.
"""
from __future__ import annotations

from typing import TypedDict


class CombinedMetrics(TypedDict, total=False):
    """Typed view of ``results/combined_metrics.json``.

    Every field here is a potential sink for a ``\\VAR{<name>}`` macro
    in the paper template.  Grouping follows the narrative order used
    by the paper itself (quality → ablations → training knobs →
    statistical uncertainty → parameter budget → optimiser settings →
    runtime → cost/energy → environment) so a reader can cross-refer
    to the table in ``report/sections/experiments.tex`` and find each
    number's definition at its declaration site.
    """
    # --- Headline F1 / NED / EM per system ---
    donut_f1: float
    donut_ned: float
    donut_em: float
    pipeline_f1: float
    pipeline_ned: float
    pipeline_em: float
    rulebased_f1: float
    rulebased_ned: float
    rulebased_gold_f1: float
    rulebased_gold_ned: float
    f1_gap: float
    assigner_delta: float
    # --- Per-field F1 for each system (drives the grouped-bar figure) ---
    donut_f1_company: float
    donut_f1_date: float
    donut_f1_address: float
    donut_f1_total: float
    rulebased_f1_company: float
    rulebased_f1_date: float
    rulebased_f1_address: float
    rulebased_f1_total: float
    # --- Training knobs echoed into the paper for reproducibility ---
    epochs_donut: int
    epochs_trocr: int
    epochs_yolo: int
    epochs_assigner: int
    batch_size: int
    lr: float
    precision: str
    label_smoothing: float
    warmup_steps: int
    yolo_img_size: int
    img_w: int
    img_h: int
    artifact_mode: str
    # --- Multi-seed aggregates (only populated when --seeds N, N>1) ---
    donut_f1_mean: float
    donut_f1_std: float
    pipeline_f1_mean: float
    pipeline_f1_std: float
    seeds_used: list[int]
    # --- Bootstrap CIs + McNemar significance (see core.statistics) ---
    donut_f1_ci_lo: float
    donut_f1_ci_hi: float
    pipeline_f1_ci_lo: float
    pipeline_f1_ci_hi: float
    mcnemar_p: float
    # --- Parameter counts + assigner training telemetry ---
    donut_params_m: float
    pipeline_params_m: float
    assigner_params_k: float
    assigner_best_epoch: int
    assigner_stopped_at: int
    assigner_best_val_loss: float
    # --- Differential LR + KD hooks (off in reported runs) ---
    lr_encoder: float
    lr_decoder: float
    kd_attn_weight: float
    kd_logits_weight: float
    # --- Pipeline diagnostics (from pipeline_metrics.json) ---
    empty_detection_fraction: float
    per_receipt_error_fraction: float
    parity_ok: bool
    # --- Hardware / runtime efficiency ---
    donut_peak_vram_gb: float
    pipeline_peak_vram_gb: float
    donut_train_minutes: float
    pipeline_train_minutes: float
    donut_samples_per_sec: float
    inference_latency_p50_ms: float
    inference_latency_p95_ms: float
    inference_latency_p99_ms: float
    # --- Cost / energy / environment ---
    donut_cost_usd: float
    pipeline_cost_usd: float
    donut_energy_kwh: float
    pipeline_energy_kwh: float
    donut_co2_kg: float
    pipeline_co2_kg: float
    gpu_model: str
    cuda_version: str
    vastai_host_id: str
