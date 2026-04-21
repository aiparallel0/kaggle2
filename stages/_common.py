"""Shared helpers used by more than one stage module.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: three short utilities that would otherwise create a circular
    import between ``stages.train`` and ``stages.eval``; kept here to
    preserve the 166-LOC cap on each stage module.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

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
