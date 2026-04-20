"""Post-eval F1 guardrails — hard-fail below bug floor."""
from __future__ import annotations

from core.errors import EvalError
from core.types import Metrics


def validate_f1(metrics: Metrics, arch: str) -> None:
    """Raise :class:`EvalError` below architecture-specific bug floor.

    Args:
        metrics: Computed :class:`Metrics` for one architecture.
        arch: Either ``"donut"`` (floor 0.50) or ``"pipeline"`` (floor > 0.0).

    F1 is stochastic (GPU, HF weights, SROIE label noise); no specific number
    can be guaranteed. Floors flag *bugs*, not underperformance. Soft-warn
    against ``expected_f1_warn`` is the caller's job — it has ``ExpConfig``.
    """
    f1 = metrics.global_f1
    if arch == "donut" and f1 < 0.50:
        raise EvalError(
            f"DONUT F1={f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), token2json list (Bug 3), "
            "or unflattened <s_sroie> wrapper (Bug 8).",
        )
    if arch == "pipeline" and f1 == 0.0:
        raise EvalError(
            "Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5) "
            "or TrOCR undertrained (Bug 6).",
        )
