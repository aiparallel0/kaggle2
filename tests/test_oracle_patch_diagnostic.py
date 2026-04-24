"""test_oracle_patch_diagnostic.py — Fix A guard.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: asserts that :func:`stages._common.oracle_patch_hybrid` still
    writes ``results/oracle_patched_fields.json`` for paper-side
    diagnostics but the return value is NOT wired back into
    ``pm.assigner`` — the eval stage has the responsibility of
    surfacing the post-patch F1 under ``oracle_patch_f1_if_applied``
    while leaving ``combined_metrics.pipeline_f1`` bound to the real
    hybrid F1.  This is the headline bug the Fix A follow-up resolved.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import load_config
from core.types import (
    ExpConfig,
    Field,
    Metrics,
    PipelineResult,
    Prediction,
    Receipt,
)
from stages._common import oracle_patch_hybrid


def _cfg(tmp_path: Path) -> ExpConfig:
    cfg = load_config("config.json")
    cfg.output_dir = str(tmp_path)
    return cfg


def _m(f1_per_field: dict[str, float]) -> Metrics:
    mean = sum(f1_per_field.values()) / max(len(f1_per_field), 1)
    return Metrics(
        global_f1=mean, global_ned=mean, global_em=mean,
        per_field_f1=dict(f1_per_field),
        per_field_ned=dict(f1_per_field),
        per_field_em=dict(f1_per_field),
    )


def test_oracle_patch_writes_diagnostic_artifact_no_regression(
    tmp_path: Path,
) -> None:
    """No regressed fields → artifact still written, returns hybrid as-is."""
    cfg = _cfg(tmp_path)
    hybrid = _m({"company": 0.90, "date": 0.90, "address": 0.90, "total": 0.90})
    rb = _m({"company": 0.70, "date": 0.70, "address": 0.70, "total": 0.70})
    pm = PipelineResult(assigner=hybrid, rulebased=hybrid)
    out = oracle_patch_hybrid(pm, rb, cfg, [])
    assert out is hybrid
    path = tmp_path / "oracle_patched_fields.json"
    assert path.exists()
    data = json.loads(path.read_text())
    assert data["regressed_fields"] == []


def test_oracle_patch_detects_regression_without_mutating_assigner(
    tmp_path: Path,
) -> None:
    """Regressed field → artifact flags it but pm.assigner is not mutated.

    This mirrors what the eval stage does: it keeps ``pm.assigner``
    bound to the hybrid's real F1 and only surfaces the post-patch
    number as a separate key.
    """
    cfg = _cfg(tmp_path)
    hybrid_pf = {"company": 0.60, "date": 0.90, "address": 0.90, "total": 0.90}
    rb_pf = {"company": 0.85, "date": 0.70, "address": 0.70, "total": 0.70}
    hybrid = _m(hybrid_pf)
    rb = _m(rb_pf)
    rcpt = Receipt(image_path="r1", fields=[Field(name="company", value="A")])
    assigner_preds = [Prediction(receipt_id="r1",
                                 fields=[Field(name="company", value="X")])]
    rulebased_preds = [Prediction(receipt_id="r1",
                                  fields=[Field(name="company", value="A")])]
    pm = PipelineResult(
        assigner=hybrid, rulebased=rb,
        assigner_preds=assigner_preds,
        rulebased_preds=rulebased_preds,
    )
    _ = oracle_patch_hybrid(pm, rb, cfg, [rcpt])
    # pm.assigner must be untouched after the diagnostic call.
    assert pm.assigner is hybrid
    assert pm.assigner.per_field_f1["company"] == pytest.approx(0.60)
    data = json.loads((tmp_path / "oracle_patched_fields.json").read_text())
    assert "company" in data["regressed_fields"]
