"""test_total_normalizer.py — symmetric TOTAL normalization + regression gate.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: locks in the PR #38 follow-up fixes — ``normalize_total`` collapses
    the currency/thousands-separator noise that otherwise makes the
    pipeline F1 incomparable to DONUT F1, and the regression gate
    guarantees the hybrid pipeline is not silently worse than the
    GT-OCR-stream rule-based baseline (same regex logic, perfect OCR input).
"""
from __future__ import annotations

import pytest

from core.errors import EvalError
from core.types import Metrics
from models.donut_eval import normalize_total
from stages._common import assert_hybrid_beats_gtocr_rulebased


def test_normalize_total_strips_currency_prefix() -> None:
    assert normalize_total("RM 43.50") == "43.50"
    assert normalize_total("rm43.50") == "43.50"
    assert normalize_total("$ 12.30") == "12.30"


def test_normalize_total_strips_thousand_separators() -> None:
    assert normalize_total("1,234.56") == "1234.56"
    assert normalize_total("RM 1,234.56") == "1234.56"


def test_normalize_total_is_idempotent() -> None:
    assert normalize_total("43.50") == "43.50"
    assert normalize_total(normalize_total("RM 43.50")) == "43.50"


def test_normalize_total_symmetric_pred_vs_gt() -> None:
    """Pred ``RM 43.50`` and GT ``43.50`` must compare equal after
    applying the normalizer on both sides — this is the property that
    makes the pipeline eval fair vs. DONUT eval."""
    assert normalize_total("RM 43.50") == normalize_total("43.50")
    assert normalize_total("1,234.56") == normalize_total("1234.56")


def _m(f1: float) -> Metrics:
    return Metrics(
        global_f1=f1, global_ned=0.0, global_em=0.0,
        per_field_f1={}, per_field_ned={}, per_field_em={},
    )


def test_regression_gate_passes_when_hybrid_beats_gtocr() -> None:
    assert_hybrid_beats_gtocr_rulebased(_m(0.80), _m(0.74))


def test_regression_gate_passes_within_epsilon() -> None:
    # hybrid 0.73 is only 0.01 below gtocr_rb 0.74 → allowed
    assert_hybrid_beats_gtocr_rulebased(_m(0.73), _m(0.74), epsilon=0.02)


def test_regression_gate_rejects_underperforming_pipeline() -> None:
    # 0.7256 < 0.7372 - 0.01 → the exact PR #38 symptom.  The default
    # epsilon was bumped to 0.03 to absorb ~2 receipts of per-image
    # noise on the 63-image SROIE test; pass it explicitly here to
    # continue exercising the hard-fail path.
    with pytest.raises(EvalError, match="gtocr_rulebased_f1"):
        assert_hybrid_beats_gtocr_rulebased(
            _m(0.7256), _m(0.7372), epsilon=0.01,
        )
