"""Post-eval F1 guardrails detecting silent F1-destroying bugs.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: hard-fails when F1 falls below architecture-specific bug floors,
    surfacing the 13 silent F1-destroying bugs documented in
    report/sections/bugs.tex (Bug 1–13).
"""
from __future__ import annotations

import json

from core.errors import EvalError
from core.types import Metrics


def validate_f1(
    metrics: Metrics,
    arch: str,
    pipeline_metrics_path: str | None = None,
) -> None:
    """Raise EvalError if F1 falls below bug floor (DONUT 0.50, pipeline >0)."""
    f1 = metrics.global_f1
    if arch == "donut" and f1 < 0.50:
        raise EvalError(
            f"DONUT F1={f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), token2json list (Bug 3), "
            "or unflattened <s_sroie> wrapper (Bug 8).",
        )
    if arch == "pipeline" and f1 == 0.0:
        diag = ""
        if pipeline_metrics_path is not None:
            try:
                with open(pipeline_metrics_path) as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    diag = (
                        f"\nDiagnostics:"
                        f" empty_detection_fraction={data.get('empty_detection_fraction')}"
                        f" per_receipt_error_fraction={data.get('per_receipt_error_fraction')}"
                        f"\nrulebased_f1={data.get('rulebased_f1')}"
                        f" assigner_f1={data.get('assigner_f1')}"
                    )
                    errtypes = data.get("receipt_error_types")
                    if errtypes:
                        diag += f"\nreceipt_error_types={errtypes}"
                    samples = data.get("receipt_error_samples")
                    if samples:
                        diag += f"\nfirst_error={samples[0]}"
            except (FileNotFoundError, json.JSONDecodeError, OSError):
                pass
        raise EvalError(
            "Pipeline F1=0.0 — YOLO imgsz mismatch (Bug 5), "
            "TrOCR undertrained (Bug 6), or stale generation_config "
            f"decoder_start_token_id (Bug 9).{diag}",
        )
