"""Tests for strategy E — 14-d distractor-aware v3 text priors.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: verifies that ``text_priors_v3`` emits a 14-d vector whose final
    five bits encode the canonical SROIE distractors (SUBTOTAL, CASH,
    CHANGE, TAX, ROUNDING) and that ``_build_priors`` correctly
    dispatches on ``n_text_priors=14``.
"""
from __future__ import annotations

import pytest

from models.attention_assign import (
    N_TEXT_PRIORS,
    N_TEXT_PRIORS_V2,
    N_TEXT_PRIORS_V3,
    text_priors_v3,
)
from models.pipeline_assign import _build_priors


def test_v3_dim_is_14_and_extends_v2() -> None:
    """v3 is v2 + 5 distractor bits, same order as text_priors_v2."""
    from models.attention_assign import text_priors_v2
    text = "TOTAL RM 43.50"
    y, last = 0.9, True
    v2 = text_priors_v2(text, y, last)
    v3 = text_priors_v3(text, y, last)
    assert len(v3) == N_TEXT_PRIORS_V3 == 14
    assert v3[: len(v2)] == v2, "v3 must preserve v2 prefix for checkpoint loading"


def test_v3_distractor_bits() -> None:
    """The five new bits each fire on exactly the right cue string."""
    cases = [
        ("SUBTOTAL 30.00", [1.0, 0.0, 0.0, 0.0, 0.0]),
        ("CASH TENDERED 50.00", [0.0, 1.0, 0.0, 0.0, 0.0]),
        ("CHANGE 6.50", [0.0, 0.0, 1.0, 0.0, 0.0]),
        ("TAX 5.00", [0.0, 0.0, 0.0, 1.0, 0.0]),
        ("ROUNDING 0.05", [0.0, 0.0, 0.0, 0.0, 1.0]),
        ("GRAND TOTAL 43.50", [0.0, 0.0, 0.0, 0.0, 0.0]),
    ]
    for text, expected in cases:
        v = text_priors_v3(text, 0.5, False)
        assert v[-5:] == expected, f"mismatch on {text!r}: got {v[-5:]}"


def test_v3_build_priors_dispatch() -> None:
    """``_build_priors`` must route n=14 to the v3 builder."""
    texts = ["ABC STORE", "SUBTOTAL 30.00", "TOTAL 43.50"]
    bboxes = [
        [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 2.0], [0.0, 2.0, 1.0, 3.0],
    ]
    out = _build_priors(texts, bboxes, N_TEXT_PRIORS_V3)
    assert len(out) == 3
    assert all(len(v) == 14 for v in out)
    # SUBTOTAL bit on row 1 only.
    assert [v[-5] for v in out] == [0.0, 1.0, 0.0]


def test_v3_still_rejects_unknown_dim() -> None:
    with pytest.raises(ValueError, match="Unsupported n_text_priors"):
        _build_priors(["x"], [[0.0, 0.0, 1.0, 1.0]], 7)


def test_all_prior_versions_coexist() -> None:
    """v1/v2/v3 dims must all be distinct and monotonically increasing."""
    assert N_TEXT_PRIORS < N_TEXT_PRIORS_V2 < N_TEXT_PRIORS_V3
    assert N_TEXT_PRIORS_V3 - N_TEXT_PRIORS_V2 == 5
