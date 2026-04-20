"""Post-eval F1 guardrails — hard-fail below bug floor, warn below expected."""
from __future__ import annotations

import logging

from core.errors import TrainError
from core.types import ExpConfig

log = logging.getLogger("kaggle2")


def validate_f1(global_f1: float, arch: str, config: ExpConfig) -> None:
    """Raise below architecture-specific bug floor; warn below expected_f1_warn.

    F1 is stochastic (GPU, HF weights, SROIE label noise); no specific number
    can be guaranteed. Floors flag *bugs*, not underperformance.
    """
    if arch == "donut" and global_f1 < 0.50:
        raise TrainError(
            f"DONUT F1={global_f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), token2json list (Bug 3), "
            "or unflattened <s_sroie> wrapper (Bug 8).",
        )
    if arch == "pipeline" and global_f1 == 0.0:
        raise TrainError(
            "Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5) "
            "or TrOCR undertrained (Bug 6).",
        )
    if global_f1 < config.expected_f1_warn:
        log.warning("%s F1=%.4f below expected_f1_warn=%.2f (not an error).",
                    arch, global_f1, config.expected_f1_warn)
