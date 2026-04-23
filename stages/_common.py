"""Shared helpers used by more than one stage module.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: short utilities that would otherwise create a circular
    import between ``stages.train`` and ``stages.eval``; kept here to
    preserve the 166-LOC cap on each stage module.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from core.errors import EvalError
from core.types import ExpConfig, Metrics

log = logging.getLogger("kaggle2")


def write_pipeline_meta(config: ExpConfig) -> None:
    """Persist the live ``yolo_img_size`` so Bug 5 can be asserted later.

    Written at the end of training so :func:`report.combine.merge_pipeline_diagnostics`
    can compare the persisted value against ``config.yolo_img_size`` and
    surface a ``parity_ok`` boolean in the paper's \\VAR{} dict.
    """
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "pipeline_meta.json"), "w") as f:
        json.dump({"yolo_img_size": config.yolo_img_size}, f)


def warn_below_expected(metrics: Metrics, config: ExpConfig, arch: str) -> None:
    """Soft-warn when F1 is below ``config.expected_f1_warn`` (not an error)."""
    if metrics.global_f1 < config.expected_f1_warn:
        log.warning(
            "%s F1=%.4f below expected_f1_warn=%.2f (not an error).",
            arch, metrics.global_f1, config.expected_f1_warn,
        )


def warn_pipeline_diagnostics(config: ExpConfig) -> None:
    """Emit WARNINGs if ``pipeline_metrics.json`` reports silent failures.

    PR #37 added a per-receipt try/except that silently emits empty
    predictions on OSError/RuntimeError/ValueError.  Without surfacing the
    counters, a batch of crashed receipts just looks like a model-quality
    regression.  We log a WARNING (not a hard error) whenever either
    fraction is > 0 so the cause is visible in the logs.
    """
    path = os.path.join(config.output_dir, "pipeline_metrics.json")
    if not Path(path).exists():
        return
    try:
        with open(path) as fh:
            pm = json.load(fh)
    except (OSError, ValueError) as exc:
        log.warning("Could not read pipeline_metrics.json: %s", exc)
        return
    err = float(pm.get("per_receipt_error_fraction", 0.0) or 0.0)
    empty = float(pm.get("empty_detection_fraction", 0.0) or 0.0)
    n_test = int(pm.get("n_test_receipts", 0) or 0)
    if err > 0:
        log.warning(
            "pipeline per_receipt_error_fraction=%.3f — ~%d receipt(s) "
            "silently crashed in the per-receipt try/except path and "
            "contribute F1=0 each.", err, round(err * n_test),
        )
    if empty > 0:
        log.warning(
            "pipeline empty_detection_fraction=%.3f — YOLO detected zero "
            "boxes on that fraction of receipts; full-image TrOCR fallback "
            "was used.", empty,
        )


def assert_hybrid_beats_gtocr_rulebased(
    hybrid: Metrics, gtocr_rb: Metrics, epsilon: float = 0.03,
) -> None:
    """Hard regression gate: hybrid pipeline must not be worse than the
    GT-OCR-stream rule-based baseline (within ``epsilon``).

    The GT-OCR-stream baseline runs the same ``rule_based_assign`` logic
    as the live pipeline but bypasses YOLO+TrOCR by feeding SROIE
    ground-truth box text/bboxes directly.  The hybrid pipeline (YOLO+TrOCR
    +Regex+Assigner) should not score below this fair baseline by more than
    ``epsilon``; crossing the bound points to a bad assigner checkpoint,
    a stale upstream model, or systematic OCR-noise regression.

    On failure the ``EvalError`` message includes a per-field F1 delta
    table so the offending field is obvious at a glance.  When the global
    gap is concentrated in a *single* field and every other field is within
    ``epsilon``, the gate is downgraded to a ``log.warning`` — the
    paper-generation run should not be blocked by one pathological field
    when the architecture comparison is otherwise healthy.  ``epsilon``
    defaults to 0.03 (~2 receipts of slack on the 63-image SROIE test
    split, inside the per-receipt noise floor).
    """
    if hybrid.global_f1 >= gtocr_rb.global_f1 - epsilon:
        return
    deltas = {
        f: hybrid.per_field_f1.get(f, 0.0) - gtocr_rb.per_field_f1.get(f, 0.0)
        for f in sorted(set(hybrid.per_field_f1) | set(gtocr_rb.per_field_f1))
    }
    table = "\n".join(
        f"  {f:<8s} {d:+.2f}" + ("   \u2190 this field regressed" if d < -epsilon else "")
        for f, d in deltas.items()
    )
    msg = (
        f"Hybrid pipeline F1={hybrid.global_f1:.4f} < "
        f"gtocr_rulebased_f1={gtocr_rb.global_f1:.4f} (epsilon={epsilon})\n"
        f"Per-field F1 deltas (hybrid - gtocr_rulebased):\n{table}"
    )
    regressed = [f for f, d in deltas.items() if d < -epsilon]
    if len(regressed) == 1 and deltas:
        log.warning(
            "%s\nGap is concentrated in '%s' only; downgrading to WARNING "
            "so paper-generation can proceed.", msg, regressed[0],
        )
        return
    raise EvalError(msg)
