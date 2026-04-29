"""FOCUS-C company-span assembler unit tests.

Exercises :func:`models.postprocess_company._company_span` on five
hand-crafted receipts covering the dominant SROIE failure modes from
``runs/<latest>/metrics/pipeline_metrics.json`` (29/347 receipts with
company F1 < 0.5):

* SDN BHD suffix on its own row,
* (REG-NO) suffix on its own row,
* boilerplate ``TAX INVOICE`` / ``CASH BILL`` prefix above the trade name,
* multi-column same-y company name (anchor only — no forward extend),
* anchor-only with no suffix line (single-line trade name).

The assembler must (a) honour the FOCUS-C ``company_pick`` argmax when
it points at a valid anchor, (b) skip boilerplate / money / date /
phone lines, (c) forward-extend through :data:`_COMPANY_SUFFIX` lines
within the contiguity gate, and (d) stop the forward walk at the first
non-suffix non-boilerplate line so unrelated text is not absorbed.
"""
from __future__ import annotations

from models.postprocess_company import _company_span


def _bbox(y1: float, y2: float) -> list[float]:
    """Helper — full-width bbox at the given vertical band (top→bottom)."""
    return [0.0, y1, 1.0, y2]


def test_sdn_bhd_suffix_extends_one_line() -> None:
    """``UNIHAKKA INTERNATIONAL`` + ``SDN BHD`` on the next row → both kept."""
    texts = [
        "UNIHAKKA INTERNATIONAL",
        "SDN BHD",
        "12 JALAN MAJU 5",
        "TOTAL RM 12.50",
    ]
    bboxes = [_bbox(0.00, 0.05), _bbox(0.06, 0.10),
              _bbox(0.20, 0.24), _bbox(0.50, 0.54)]
    bp = [False, False, False, False]
    picks, value = _company_span(texts, bboxes, bp, anchor_idx=0)
    assert picks == [0, 1]
    assert value == "UNIHAKKA INTERNATIONAL SDN BHD"


def test_reg_no_suffix_extends_two_lines() -> None:
    """``GROCER MART`` + ``SDN BHD`` + ``(123456-A)`` → all three kept."""
    texts = [
        "GROCER MART",
        "SDN BHD",
        "(123456-A)",
        "12 JALAN MAJU 5",
    ]
    bboxes = [_bbox(0.00, 0.04), _bbox(0.05, 0.09),
              _bbox(0.10, 0.14), _bbox(0.30, 0.34)]
    bp = [False, False, False, False]
    picks, value = _company_span(texts, bboxes, bp, anchor_idx=0)
    assert picks == [0, 1, 2]
    assert "GROCER MART" in value
    assert "SDN BHD" in value
    assert "(123456-A)" in value


def test_boilerplate_prefix_skipped_anchor_falls_through() -> None:
    """``TAX INVOICE / CASH BILL`` above the trade name must NOT anchor;
    the assembler skips them and selects the merchant line below.
    """
    texts = [
        "TAX INVOICE",            # 0 — boilerplate
        "CASH BILL",              # 1 — boilerplate
        "UNIHAKKA INTERNATIONAL", # 2 — true anchor
        "SDN BHD",                # 3 — suffix
        "47000 SUNGAI BULOH",     # 4 — address
    ]
    bboxes = [_bbox(0.00, 0.04), _bbox(0.05, 0.09),
              _bbox(0.10, 0.14), _bbox(0.15, 0.19),
              _bbox(0.30, 0.34)]
    bp = [False, False, False, False, False]
    # Pretend the FOCUS-C head wrongly anchored on the boilerplate line —
    # the assembler must reject it via the boilerplate-keyword gate and
    # fall back to the topmost qualifying line.
    picks, value = _company_span(texts, bboxes, bp, anchor_idx=0)
    assert picks == [2, 3]
    assert "TAX INVOICE" not in value
    assert "CASH BILL" not in value
    assert value.startswith("UNIHAKKA INTERNATIONAL")


def test_multi_column_same_y_anchor_only_no_forward() -> None:
    """Two columns at the same y — the anchor is honoured but no forward
    extension fires because the next line below is not a suffix.
    """
    texts = [
        "ACME TRADING CO",       # 0 — anchor candidate
        "BRANCH 12",             # 1 — same-y companion column (not a suffix)
        "12 JALAN MAJU 5",       # 2 — address
    ]
    bboxes = [_bbox(0.00, 0.04), _bbox(0.00, 0.04),
              _bbox(0.10, 0.14)]
    bp = [False, False, False]
    picks, value = _company_span(texts, bboxes, bp, anchor_idx=0)
    # ``BRANCH 12`` does NOT match _COMPANY_SUFFIX → forward walk stops.
    assert picks == [0]
    assert value == "ACME TRADING CO"


def test_anchor_only_no_suffix() -> None:
    """A single-line merchant trade name with no SDN BHD / (REG-NO)
    suffix returns just the anchor.
    """
    texts = [
        "BLUE OCEAN MART",
        "12 JALAN MAJU 5",
        "47000 SUNGAI BULOH",
    ]
    bboxes = [_bbox(0.00, 0.04), _bbox(0.10, 0.14), _bbox(0.20, 0.24)]
    bp = [False, False, False]
    picks, value = _company_span(texts, bboxes, bp, anchor_idx=0)
    assert picks == [0]
    assert value == "BLUE OCEAN MART"


def test_priors_v4_boilerplate_flag_skips_line() -> None:
    """When ``priors_v4[is_company_boilerplate]`` is set on a line that
    would otherwise match :data:`_COMPANY_SUFFIX`, the forward walk
    SKIPS that line (does not stop) — this is the network's learned
    boilerplate signal compounding with the regex.
    """
    texts = [
        "GROCER MART",
        "SDN BHD",          # 1 — suffix-shaped but flagged by network
        "(123456-A)",       # 2 — should still be picked
        "12 JALAN MAJU 5",
    ]
    bboxes = [_bbox(0.00, 0.04), _bbox(0.05, 0.09),
              _bbox(0.10, 0.14), _bbox(0.30, 0.34)]
    bp = [False, True, False, False]  # network flagged line 1 as boilerplate
    picks, value = _company_span(texts, bboxes, bp, anchor_idx=0)
    # Anchor (0) kept; line 1 skipped (boilerplate prior); line 2 picked.
    assert 0 in picks
    assert 2 in picks
    assert 1 not in picks
    assert "GROCER MART" in value
    assert "(123456-A)" in value


def test_empty_inputs_return_empty() -> None:
    """Degenerate input — empty texts / mismatched lengths → ``([], "")``."""
    assert _company_span([], [], []) == ([], "")
    assert _company_span(["a"], [], []) == ([], "")
    assert _company_span(["a"], [_bbox(0.0, 0.1)], []) == ([], "")


def test_invalid_anchor_falls_back_to_top_qualifying_line() -> None:
    """``anchor_idx`` pointing at a money/date line is rejected; the
    assembler falls back to the topmost qualifying anchor.
    """
    texts = [
        "ACME TRADING CO",        # 0 — true anchor
        "12 JALAN MAJU 5",        # 1
        "TOTAL RM 12.50",         # 2 — money line, NOT an anchor
    ]
    bboxes = [_bbox(0.00, 0.04), _bbox(0.10, 0.14), _bbox(0.30, 0.34)]
    bp = [False, False, False]
    picks, value = _company_span(texts, bboxes, bp, anchor_idx=2)
    # Index 2 is a money line → rejected; falls back to topmost anchor (0).
    assert picks == [0]
    assert value == "ACME TRADING CO"
