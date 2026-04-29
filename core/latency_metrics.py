"""Inference-time latency / throughput diagnostics.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: reduce a list of per-request latency timings (milliseconds) —
    collected by ``app/predict.py`` / ``models/eval_pipeline.py`` /
    ``models/donut_eval.py`` over a burst of synthetic requests — into
    the latency profile surfaced in the paper's Table IV: cold-start
    time, p50/p95/p99, max, mean, and batch-1 / batch-8 throughput.
"""
from __future__ import annotations

from collections.abc import Sequence

from core.schemas import SCHEMA_VERSIONS, LatencyDiagnostics


def _percentile(values: Sequence[float], q: float) -> float:
    """Inclusive linear-interp percentile (q in [0, 1])."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def compute_latency_diagnostics(
    system: str,
    cold_start_ms: float,
    hot_latencies_ms: Sequence[float],
    batch1_throughput: float = 0.0,
    batch8_throughput: float = 0.0,
) -> LatencyDiagnostics:
    """Aggregate a hot-path timing burst into the latency JSON schema."""
    if not hot_latencies_ms:
        return LatencyDiagnostics(
            schema_version=SCHEMA_VERSIONS["LatencyDiagnostics"],
            system=system,
            cold_start_ms=float(cold_start_ms),
            throughput_batch1=float(batch1_throughput),
            throughput_batch8=float(batch8_throughput),
        )
    return LatencyDiagnostics(
        schema_version=SCHEMA_VERSIONS["LatencyDiagnostics"],
        system=system,
        cold_start_ms=float(cold_start_ms),
        p50_ms=_percentile(hot_latencies_ms, 0.5),
        p95_ms=_percentile(hot_latencies_ms, 0.95),
        p99_ms=_percentile(hot_latencies_ms, 0.99),
        max_ms=max(hot_latencies_ms),
        mean_ms=sum(hot_latencies_ms) / len(hot_latencies_ms),
        throughput_batch1=float(batch1_throughput),
        throughput_batch8=float(batch8_throughput),
    )
