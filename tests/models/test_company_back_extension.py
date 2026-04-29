"""Unit tests for the multi-line backward extension in
:func:`models.focus_pipeline._company_anchor_filter`.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: lock the multi-line back-walk behaviour driven by the Fig.~8
    error decomposition: ``company`` is dominated by ``wrong_span``
    misses (the FOCUS-C span head picks the SDN BHD anchor cleanly
    but the registered name above is dropped).  The pre-PR filter
    extended only one line back, with an alpha+space-only character
    set, so receipts whose registered name spans 3–5 lines (or
    contains ``&``/``.``/digits) under-extended chronically.
"""
from __future__ import annotations

from models.focus_pipeline import _company_anchor_filter


def _bbox(n: int) -> list[list[float]]:
    """Helper: stacked top-to-bottom unit boxes for ``n`` lines."""
    return [[0.0, float(i), 1.0, float(i + 1)] for i in range(n)]


def test_legacy_single_line_back_preserved() -> None:
    """The pre-existing 1-line, alpha+space back-extension still fires."""
    texts = ["KEDAI UBAT", "HONG NING SDN BHD"]
    out = _company_anchor_filter([0, 1], texts, _bbox(2))
    assert out == [0, 1]


def test_back_extension_with_ampersand() -> None:
    """X00000014-style: the registered name uses ``&`` and the legacy
    alpha+space-only filter dropped the continuation line."""
    texts = ["KEDAI UBAT", "& RUNCIT", "HONG NING SDN BHD"]
    out = _company_anchor_filter([0, 1, 2], texts, _bbox(3))
    assert out == [0, 1, 2]


def test_back_extension_three_lines() -> None:
    """X00000067/X00000104/X00000149-style: the registered name spans
    three lines above the SDN BHD anchor."""
    texts = ["PASARAYA", "BORONG", "PINTAR", "SDN BHD"]
    out = _company_anchor_filter([0, 1, 2, 3], texts, _bbox(4))
    assert out == [0, 1, 2, 3]


def test_back_extension_capped_at_four() -> None:
    """Cap at 4 back so a noisy top-of-receipt header band cannot
    pollute the company span on a clean-line receipt."""
    texts = ["LINE5", "LINE4", "LINE3", "LINE2", "LINE1", "COMPANY SDN BHD"]
    out = _company_anchor_filter([0, 1, 2, 3, 4, 5], texts, _bbox(6))
    assert out == [1, 2, 3, 4, 5]


def test_back_extension_stops_at_header() -> None:
    """The receipt-header regex (``TAX INVOICE`` / ``RECEIPT`` / etc.)
    breaks the backward walk so the anchor remains alone."""
    texts = ["TAX INVOICE", "PASARAYA BORONG SDN BHD"]
    out = _company_anchor_filter([0, 1], texts, _bbox(2))
    assert out == [1]


def test_back_extension_stops_at_lowercase_metadata() -> None:
    """Lowercase / mixed-case metadata (``receipt time 14:30``) is not
    a trade-name continuation — break the walk."""
    texts = ["receipt time 14:30", "PASARAYA SDN BHD"]
    out = _company_anchor_filter([0, 1], texts, _bbox(2))
    assert out == [1]


def test_back_extension_stops_at_reg_id() -> None:
    """A registration-ID line (``(123456-W)``) terminates the walk so
    the SDN BHD anchor never inherits the registration boilerplate."""
    texts = ["(123456-W)", "PASARAYA SDN BHD"]
    out = _company_anchor_filter([0, 1], texts, _bbox(2))
    assert out == [1]


def test_no_anchor_returns_topmost_clean() -> None:
    """When no SDN BHD / BERHAD / ENTERPRISE anchor is present, the
    function falls back to the topmost clean line — preserved from
    the legacy implementation."""
    texts = ["RESTORAN WAN SHENG", "JALAN TEMENGGUNG"]
    out = _company_anchor_filter([0, 1], texts, _bbox(2))
    assert out == [0]
