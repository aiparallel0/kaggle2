"""TypedDict schemas for every run-generated JSON artefact.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: concrete ``TypedDict`` definitions for the JSON sidecars produced
    under ``runs/<run_id>/metrics/`` and ``runs/<run_id>/env/``.  These
    are the reviewer-facing contract between the metric producers
    (``core/metrics_*.py``, ``models/*_diagnose.py``) and the LaTeX
    ``\\VAR{}`` substitution in :mod:`report.inject`.  Every new JSON
    emitted at runtime MUST reference a schema in this module so the
    mypy-as-test-suite contract stays meaningful.

Keep this file flat (no nested TypedDicts across modules) so reviewers
can read the full surface in one place.  Schema version bumps are
recorded in ``SCHEMA_VERSIONS`` at the bottom.
"""
from __future__ import annotations

from typing import TypedDict

# ---------------------------------------------------------------------
# Env / reproducibility snapshot — written once per run by
# :mod:`core.env_snapshot` into ``<run_dir>/env/``.
# ---------------------------------------------------------------------


class HostInfo(TypedDict, total=False):
    """Static host description — CPU, RAM, GPU, driver, CUDA."""

    hostname: str
    platform: str
    cpu_model: str
    cpu_count: int
    ram_gb: float
    gpu_model: str
    gpu_count: int
    gpu_vram_gb: float
    driver_version: str
    cuda_version: str
    torch_version: str
    python_version: str


class EnvSnapshot(TypedDict, total=False):
    """Top-level env summary written to ``env/hostinfo.json``."""

    schema_version: int
    run_id: str
    run_utc: str
    git_sha: str
    config_sha256: str
    seed: int
    host: HostInfo


# ---------------------------------------------------------------------
# Per-architecture diagnostics — one JSON per architecture under
# ``<run_dir>/metrics/``.  ``total=False`` so back-compat stays trivial
# when a newer schema version adds keys.
# ---------------------------------------------------------------------


class YoloDiagnostics(TypedDict, total=False):
    """``metrics/yolo_metrics.json`` — detection diagnostics."""

    schema_version: int
    map50: float
    map5095: float
    per_class_ap: dict[str, float]
    iou_median: float
    iou_mean: float
    boxes_per_receipt_mean: float
    boxes_per_receipt_median: float
    p_at_0_25: float
    r_at_0_25: float
    pr_curve_precision: list[float]
    pr_curve_recall: list[float]


class TrocrDiagnostics(TypedDict, total=False):
    """``metrics/trocr_metrics.json`` — TrOCR CER/WER diagnostics."""

    schema_version: int
    cer_mean: float
    cer_total: float
    wer_mean: float
    cer_per_field: dict[str, float]
    wer_per_field: dict[str, float]
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float


class AssignerDiagnostics(TypedDict, total=False):
    """``metrics/assigner_diagnostics.json`` — attention assigner diagnostics."""

    schema_version: int
    entropy_per_field: dict[str, float]
    attention_peak_sharpness: dict[str, float]
    ece: float
    mce: float
    brier: float
    top1_acc: float
    top3_acc: float
    top5_acc: float
    level1_acc: float
    level2_acc: float
    prior_posterior_kl: float


class DonutDiagnostics(TypedDict, total=False):
    """``metrics/donut_diagnostics.json`` — DONUT generation diagnostics."""

    schema_version: int
    invalid_json_rate: float
    mean_logprob: float
    token_acc: float
    attn_entropy_mean: float
    gen_len_p50: float
    gen_len_p95: float
    special_token_acc: float
    beam_agreement_rate: float


class LatencyDiagnostics(TypedDict, total=False):
    """``metrics/latency_<system>.json`` — inference-time profile."""

    schema_version: int
    system: str
    cold_start_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float
    mean_ms: float
    throughput_batch1: float
    throughput_batch8: float


class CostDiagnostics(TypedDict, total=False):
    """``metrics/cost_<stage>.json`` — derived wall-clock + $/kWh."""

    schema_version: int
    stage: str
    wall_seconds: float
    usd: float
    energy_wh: float
    gpu_model: str


# ---------------------------------------------------------------------
# Schema version registry — reviewers track ``schema_version`` across
# runs so stale tooling fails loudly rather than silently degrading.
# Bump any time a producer emits a new key the paper depends on.
# ---------------------------------------------------------------------

SCHEMA_VERSIONS: dict[str, int] = {
    "EnvSnapshot": 1,
    "YoloDiagnostics": 1,
    "TrocrDiagnostics": 1,
    "AssignerDiagnostics": 1,
    "DonutDiagnostics": 1,
    "LatencyDiagnostics": 1,
    "CostDiagnostics": 1,
}
