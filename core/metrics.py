"""Shared metric computation: token-F1, NED, EM — used by both eval modules.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: canonical implementations of Levenshtein edit distance, token-F1,
    and the :func:`compute_metrics` reducer over an :class:`EvalBundle`.
    Also re-exports the ``core.statistics`` helpers so callers can write
    ``from core.metrics import bootstrap_ci`` without a second import.
"""
from __future__ import annotations

from typing import TypedDict

from core.statistics import bootstrap_ci as bootstrap_ci  # noqa: PLC0414
from core.statistics import mcnemar as mcnemar  # noqa: PLC0414
from core.statistics import ned_buckets as ned_buckets  # noqa: PLC0414
from core.types import EvalBundle, Metrics


class CombinedMetrics(TypedDict, total=False):
    """Typed view of ``results/combined_metrics.json``.

    Single source of truth for every number surfaced in the paper via
    ``\\VAR{}`` substitution. ``total=False`` because keys are populated
    incrementally (e.g. multi-seed mean/std only appear when the
    ``--seeds`` harness aggregates more than one run).
    """
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
    donut_f1_company: float
    donut_f1_date: float
    donut_f1_address: float
    donut_f1_total: float
    rulebased_f1_company: float
    rulebased_f1_date: float
    rulebased_f1_address: float
    rulebased_f1_total: float
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
    donut_f1_mean: float
    donut_f1_std: float
    pipeline_f1_mean: float
    pipeline_f1_std: float
    seeds_used: list[int]
    # --- Bootstrap CIs + significance ---
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
    # --- Hardware / efficiency ---
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


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def ned(a: str, b: str) -> float:
    """Normalised Edit Distance: 1.0 = identical, 0.0 = completely different."""
    if not a and not b:
        return 1.0
    dist = edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


def token_f1(a: str, b: str) -> float:
    """Token-level F1 between ground-truth *a* and prediction *b*."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = ta & tb
    p = len(common) / len(tb)
    r = len(common) / len(ta)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_metrics(bundle: EvalBundle) -> Metrics:
    """Compute per-field and global F1 / NED / EM from an :class:`EvalBundle`."""
    pf1: dict[str, list[float]] = {f: [] for f in bundle.fields}
    pned: dict[str, list[float]] = {f: [] for f in bundle.fields}
    pem: dict[str, list[float]] = {f: [] for f in bundle.fields}
    for pred, rec in zip(bundle.predictions, bundle.receipts, strict=True):
        gt = {fld.name.lower(): fld.value.lower() for fld in rec.fields}
        pr = {fld.name.lower(): fld.value.lower() for fld in pred.fields}
        for f in bundle.fields:
            g = gt.get(f, "")
            p = pr.get(f, "")
            pem[f].append(1.0 if g == p else 0.0)
            pned[f].append(ned(g, p))
            pf1[f].append(token_f1(g, p))
    per_f1 = {f: sum(v) / len(v) for f, v in pf1.items() if v}
    per_ned = {f: sum(v) / len(v) for f, v in pned.items() if v}
    per_em = {f: sum(v) / len(v) for f, v in pem.items() if v}
    g_f1 = sum(per_f1.values()) / len(per_f1) if per_f1 else 0.0
    g_ned = sum(per_ned.values()) / len(per_ned) if per_ned else 0.0
    g_em = sum(per_em.values()) / len(per_em) if per_em else 0.0
    return Metrics(
        global_f1=g_f1, global_ned=g_ned, global_em=g_em,
        per_field_f1=per_f1, per_field_ned=per_ned, per_field_em=per_em,
    )
