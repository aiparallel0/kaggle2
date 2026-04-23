"""Regression tests for the train/eval priors contract mismatch.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: verify that pipeline_assign._build_priors dispatches on the assigner's
    n_text_priors (6→v1, 9→v2) and that the v2 construction matches the
    training-time signals in models/assigner_data (y_norm, is_last_money_line).
"""
from __future__ import annotations

import pytest

from models.attention_assign import N_TEXT_PRIORS, N_TEXT_PRIORS_V2
from models.pipeline_assign import _build_priors


def test_build_priors_v1_dim() -> None:
    """With n_priors=6 (v1), every region gets a 6-d vector."""
    texts = ["ABC Cafe", "TOTAL: 12.00", "Kuala Lumpur"]
    bboxes = [[0.0, 0.0, 10.0, 1.0], [0.0, 2.0, 10.0, 3.0], [0.0, 4.0, 10.0, 5.0]]
    out = _build_priors(texts, bboxes, N_TEXT_PRIORS)
    assert len(out) == len(texts)
    assert all(len(v) == N_TEXT_PRIORS for v in out)


def test_build_priors_v2_dim_and_signals() -> None:
    """With n_priors=9 (v2), vectors carry y_norm + is_last_money_line."""
    texts = ["ABC Cafe", "ITEM 5.00", "TOTAL 12.00"]
    bboxes = [
        [0.0, 0.0, 10.0, 1.0],
        [0.0, 2.0, 10.0, 3.0],
        [0.0, 4.0, 10.0, 5.0],  # last MONEY_RE hit, max y-bottom
    ]
    out = _build_priors(texts, bboxes, N_TEXT_PRIORS_V2)
    assert len(out) == len(texts)
    assert all(len(v) == N_TEXT_PRIORS_V2 for v in out)
    # is_last_money_line is element [-2]; only the last row should be 1.0.
    flags = [v[-2] for v in out]
    assert flags == [0.0, 0.0, 1.0]
    # y_norm is element [-1]; last row's y_bottom == max_y ⇒ 1.0.
    y_norms = [v[-1] for v in out]
    assert y_norms[-1] == pytest.approx(1.0)
    assert y_norms[0] < y_norms[1] < y_norms[2]


def test_build_priors_no_money_marks_none_last() -> None:
    """With no money match, is_last_money_line stays 0 for every region."""
    texts = ["Hello", "World"]
    bboxes = [[0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 1.0, 2.0]]
    out = _build_priors(texts, bboxes, N_TEXT_PRIORS_V2)
    assert [v[-2] for v in out] == [0.0, 0.0]


def test_build_priors_rejects_unsupported_dim() -> None:
    """Unknown n_text_priors fails loudly instead of silently zero-padding."""
    with pytest.raises(ValueError, match="Unsupported n_text_priors"):
        _build_priors(["a"], [[0.0, 0.0, 1.0, 1.0]], 7)
