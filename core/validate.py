"""Post-eval F1 guardrails detecting silent F1-destroying bugs.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: hard-fails when F1 falls below architecture-specific bug floors,
    surfacing the 13 silent F1-destroying bugs documented in
    report/sections/bugs.tex (Bug 1–13).
"""
from __future__ import annotations

from core.errors import EvalError
from core.types import Metrics


def validate_f1(metrics: Metrics, arch: str) -> None:
    """Raise EvalError if F1 falls below bug floor (DONUT 0.50, pipeline >0)."""
    f1 = metrics.global_f1
    if arch == "donut" and f1 < 0.50:
        raise EvalError(
            f"DONUT F1={f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), token2json list (Bug 3), "
            "or unflattened <s_sroie> wrapper (Bug 8).",
        )
    if arch == "pipeline" and f1 == 0.0:
        raise EvalError(
            "Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5), "
            "TrOCR undertrained (Bug 6), or stale generation_config "
            "decoder_start_token_id (Bug 9).",
        )
