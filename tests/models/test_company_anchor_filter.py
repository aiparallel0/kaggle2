"""FOCUS-C — _company_anchor_filter unit tests.

Tests the anchor-based company-span filtering logic from
:func:`models.focus_pipeline._company_anchor_filter`.
"""
from __future__ import annotations

from models.focus_pipeline import (
    _COMPANY_ANCHOR,
    _COMPANY_REG_ID_RE,
    _company_anchor_filter,
    _is_company_boundary,
)


def test_is_company_boundary_detects_registration() -> None:
    """Company registration IDs are boundaries."""
    assert _is_company_boundary("(123456-W)") is True


def test_company_anchor_regex_matches_common_suffixes() -> None:
    """_COMPANY_ANCHOR matches SDN BHD and similar suffixes."""
    assert _COMPANY_ANCHOR.search("GROCER MART SDN BHD") is not None
    assert _COMPANY_ANCHOR.search("ABC TRADING SDN. BHD.") is not None
    assert _COMPANY_ANCHOR.search("COMPANY ENTERPRISE") is not None


def test_company_reg_id_regex_matches() -> None:
    """_COMPANY_REG_ID_RE matches registration patterns."""
    assert _COMPANY_REG_ID_RE.search("(123456-W)") is not None
    assert _COMPANY_REG_ID_RE.search("CO REG NO 789012-X") is not None


def test_company_anchor_filter_finds_anchor_and_extends() -> None:
    """Filter picks anchor row, extends backward/forward appropriately."""
    texts = [
        "GROCER",
        "MART SDN BHD",
        "TEL: 03-12345678",  # Phone line is a boundary
    ]
    bboxes = [[0, 0, 100, 20], [0, 25, 100, 45], [0, 50, 100, 70]]
    picks = [0, 1, 2]
    result = _company_anchor_filter(picks, texts, bboxes)
    # Anchor at "MART SDN BHD" (idx 1), extend back to "GROCER" (idx 0)
    # Phone line (idx 2) is a boundary, excluded
    assert result == [0, 1]


def test_company_anchor_filter_fallback_when_no_anchor() -> None:
    """No anchor keyword in picks → return first clean pick."""
    texts = ["HELLO WORLD", "SOME TEXT"]
    bboxes = [[0, 0, 100, 20], [0, 25, 100, 45]]
    picks = [0, 1]
    result = _company_anchor_filter(picks, texts, bboxes)
    # No anchor found in clean, fallback to first clean pick
    assert result == [0]


def test_company_anchor_filter_empty_picks() -> None:
    """Empty picks input → empty result."""
    result = _company_anchor_filter([], ["foo"], [[0, 0, 1, 1]])
    assert result == []
