"""test_oracle_patch_noop.py — regression guard for the oracle patcher.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: locks in the *diagnostic-only* contract of
    :func:`stages._common.oracle_patch_hybrid`.  The original clobbering
    bug ("pipeline F1 0.80 → 0.58") appeared when post-patch metrics
    were substituted into ``combined_metrics.pipeline_f1``.  These
    tests assert two complementary invariants:

      1. When rulebased-on-noisy-TrOCR is *worse* than the assigner on
         every field, the caller-visible ``pm.assigner`` stays put.
         The patcher may still write its diagnostic JSON, but it
         MUST NOT mutate the headline metrics.
      2. When the post-patch global F1 would regress, the patcher
         emits a WARNING so the regression is visible in
         ``kaggle2_pipeline.log`` (without changing behaviour).

    Together these would have caught the 0.80→0.58 bug at CI time.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.config import load_config
from core.types import (
    EvalBundle,
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


def _metrics(per_field: dict[str, float]) -> Metrics:
    mean = sum(per_field.values()) / max(len(per_field), 1)
    return Metrics(
        global_f1=mean, global_ned=mean, global_em=mean,
        per_field_f1=dict(per_field),
        per_field_ned=dict(per_field),
        per_field_em=dict(per_field),
    )


def test_headline_f1_not_mutated_when_rulebased_worse(tmp_path: Path) -> None:
    """Assigner strictly better on every field ⇒ pm.assigner untouched.

    Because no field regresses beyond ``epsilon``, :func:`oracle_patch_hybrid`
    returns the unmodified assigner metrics and writes a diagnostic
    JSON showing ``regressed_fields == []``.  The headline F1 the
    caller ultimately reports is therefore guaranteed to equal the
    pre-patch assigner F1.
    """
    cfg = _cfg(tmp_path)
    hybrid_pf = {"company": 0.85, "date": 0.87, "address": 0.70, "total": 0.80}
    rb_pf = {"company": 0.60, "date": 0.55, "address": 0.50, "total": 0.40}
    hybrid = _metrics(hybrid_pf)
    rb = _metrics(rb_pf)
    pm = PipelineResult(assigner=hybrid, rulebased=rb)
    pre_f1 = pm.assigner.global_f1

    out = oracle_patch_hybrid(pm, rb, cfg, [])

    # Diagnostic-only contract: assigner metrics unchanged.
    assert pm.assigner is hybrid
    assert pm.assigner.global_f1 == pre_f1
    # Return value is the assigner metrics verbatim (no regression detected).
    assert out is hybrid
    data = json.loads((tmp_path / "oracle_patched_fields.json").read_text())
    assert data["regressed_fields"] == []


def test_warns_when_postpatch_f1_regresses(
    tmp_path: Path, caplog: logging.LogRecord,
) -> None:
    """Patch fires but post-patch F1 < pre-patch F1 ⇒ WARNING logged.

    Reproduces the 0.80→0.58 diagnostic at miniature scale: one field
    regresses by > epsilon (triggers the patch), but the rulebased
    prediction on the live receipt is *wrong* so substituting it
    decreases global F1.  The contract is that the returned post-patch
    metrics reflect the regression and a WARNING is emitted — but
    ``pm.assigner`` (the headline metrics) remains untouched.
    """
    cfg = _cfg(tmp_path)
    # 'address' regresses on paper metrics → the patcher will try to patch.
    hybrid_pf = {"company": 0.90, "address": 0.60}
    rb_pf = {"company": 0.70, "address": 0.90}
    hybrid = _metrics(hybrid_pf)
    rb = _metrics(rb_pf)
    # But on the actual receipts, the rulebased 'address' value is WRONG
    # (it's "NOT THE ADDRESS") while the assigner's value is correct.
    # compute_metrics will therefore score the patched prediction LOWER
    # than the unpatched one — exactly the 0.80→0.58 failure mode.
    gt_addr = "123 MAIN ST"
    rcpt = Receipt(
        image_path=Path("r1"),
        fields=[Field(name="address", value=gt_addr)],
    )
    assigner_preds = [Prediction(
        receipt_id="r1", fields=[Field(name="address", value=gt_addr)],
    )]
    rulebased_preds = [Prediction(
        receipt_id="r1",
        fields=[Field(name="address", value="NOT THE ADDRESS")],
    )]
    pm = PipelineResult(
        assigner=hybrid, rulebased=rb,
        assigner_preds=assigner_preds,
        rulebased_preds=rulebased_preds,
    )
    cfg.fields = ["address"]
    pre_f1 = pm.assigner.global_f1

    with caplog.at_level(logging.WARNING, logger="kaggle2"):
        patched = oracle_patch_hybrid(pm, rb, cfg, [rcpt])

    # 1. Headline metrics untouched — the diagnostic-only contract.
    assert pm.assigner is hybrid
    assert pm.assigner.global_f1 == pre_f1
    # 2. Post-patch F1 is strictly lower (rulebased prediction is wrong).
    assert patched.global_f1 < pre_f1
    # 3. A WARNING was emitted naming the regression.
    assert any(
        "post-patch F1" in rec.getMessage() and rec.levelno == logging.WARNING
        for rec in caplog.records
    ), f"expected regression WARNING, got: {[r.getMessage() for r in caplog.records]}"
    # 4. The diagnostic JSON records the negative delta honestly.
    data = json.loads((tmp_path / "oracle_patched_fields.json").read_text())
    assert "post_patch_global_f1_delta" in data
    assert data["post_patch_global_f1_delta"] < 0


def test_eval_bundle_import_available() -> None:
    """Sanity: EvalBundle is importable (used by the patcher internally)."""
    assert EvalBundle is not None
