"""Tests for strategies L (additive attn scoring) and H (confidence gating).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: guards the inference-side assigner fixes from the plan:
    - strategy L: ``_score_money`` blends ``log(attn+ε)`` additively so a
      confident attention peak only tilts ties, never overwhelms a clean
      ``_TOTAL_STRONG`` match;
    - strategy H: ``_is_attn_diffuse`` + the relaxed override margin let
      the rule scorer override a well-formed learned money value when
      the learned attention row is flat (uniform / low top-1−top-2).
"""
from __future__ import annotations

from models.pipeline_consensus import (
    _ATTN_DIFFUSE_ENTROPY,
    _attn_entropy,
    _attn_margin,
    _is_attn_diffuse,
    _refine_total,
)


def _bb(y1: float, y2: float) -> list[float]:
    return [0.0, y1, 1.0, y2]


def test_attn_entropy_peaked_vs_uniform() -> None:
    """Normalised entropy: one-hot → 0, uniform → 1."""
    assert _attn_entropy([1.0, 0.0, 0.0, 0.0]) == 0.0
    assert abs(_attn_entropy([0.25] * 4) - 1.0) < 1e-6


def test_attn_margin_peaked_vs_uniform() -> None:
    """Top1−Top2 margin: one-hot ≈ 1, uniform ≈ 0."""
    assert abs(_attn_margin([1.0, 0.0, 0.0, 0.0]) - 1.0) < 1e-6
    assert abs(_attn_margin([0.25] * 4)) < 1e-6


def test_is_attn_diffuse_flags_flat_attention() -> None:
    """A ~uniform attention row is flagged; a sharp one is not."""
    assert _is_attn_diffuse([0.25] * 4) is True
    assert _is_attn_diffuse(None) is True
    assert _is_attn_diffuse([]) is True
    # Sharp distribution with entropy < 0.80 and margin > 0.05.
    sharp = [0.9, 0.05, 0.03, 0.02]
    assert _attn_entropy(sharp) < _ATTN_DIFFUSE_ENTROPY
    assert _is_attn_diffuse(sharp) is False


def test_additive_attn_does_not_flip_strong_keyword_match() -> None:
    """Strategy L — a confident attention peak on SUBTOTAL does NOT
    beat a clean ``TOTAL`` keyword match. Log-attn blend weight is
    bounded (0.5) so the +4 ``_TOTAL_STRONG`` term still dominates."""
    texts = [
        "ITEM 1 5.00",           # 0
        "SUBTOTAL 30.00",        # 1  (attention says this)
        "TAX 0.00",              # 2
        "GRAND TOTAL RM 43.50",  # 3
        "THANK YOU",             # 4
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    # Assigner confidently picked SUBTOTAL (line 1), but GRAND TOTAL has
    # a decisive keyword signal.  Refiner must still flip to 43.50.
    attn = [0.02, 0.90, 0.02, 0.04, 0.02]
    assert _refine_total("30.00", texts, bboxes, attn_row=attn) == "43.50"


def test_diffuse_attention_relaxes_override_margin() -> None:
    """Strategy H — when learned attention is flat, the rule scorer is
    allowed to override a well-formed learned money pick on weaker
    (<2.0) evidence.  The conservative default would keep the learned
    pick; the diffuse-attention relaxation flips it."""
    # Two money lines with only mild positive signals on line 1 (last
    # money, +1.5) and no keyword.  Rule score diff ≈ 1.8 — below the
    # conservative 2.0 margin, above the diffuse 0.5 margin.
    texts = [
        "ACME STORE 10.00",  # 0 — learned picked here
        "25.00",             # 1 — last money line → best rule score
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    flat_attn = [0.45, 0.55]  # diffuse → relaxed margin
    assert _refine_total("10.00", texts, bboxes, attn_row=flat_attn) == "25.00"
    # Peaked attention on the learned pick → that line wins the scored
    # ranking (best == learned), so no override fires regardless of
    # margin.  This is the "confident assigner is trusted" half of H.
    sharp_attn = [0.95, 0.05]
    assert _refine_total("10.00", texts, bboxes, attn_row=sharp_attn) == "10.00"
