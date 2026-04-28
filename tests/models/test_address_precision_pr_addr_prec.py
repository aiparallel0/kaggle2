"""PR-ADDR-PREC — post-FOCUS span boundary tightening.

Targets the precision-bottleneck pattern in ``address_mismatches.json``:
company headers (``BHD``, ``INTERNATIONAL``, ``SDN``), invoice/cashier
metadata (``INV NO``, ``CASH``, ``RECEIPT``, ``DOC NO``), tax/GST IDs,
and trailing 1-2-char OCR fragments leaking into the predicted address
span.  Each test mirrors a real failure receipt from the live miss
table; assertions check that the contaminated tokens are stripped
without losing the legitimate postcode + city tail.
"""
from __future__ import annotations

import pytest

from models.consensus import _address_span, _is_addr_boundary
from models.postprocess_address import normalize_address_focus
from models.rule_regex import _ADDR_COMPANY_HEADER, _ADDR_TERMINATOR

# ---------------------------------------------------------------------------
# Boundary regex coverage — pin the keyword set documented in the PR brief.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "INV NO 1075214",
    "INV. NO 1075214",
    "INVOICE NO ABC123",
    "TAX INVOICE",
    "RECEIPT",
    "RECEIPT NO 0001",
    "CASHIER: SARAH",
    "CASH SALE",
    "CASH",
    "TABLE 5",
    "TABLE NO 12",
    "ORDER NO 7",
    "DOC NO 101",
    "DOC #",
    "COVER",
    "WAITER: JOHN",
    "BILL",
    "BILL TO",
    "CREDIT NOTE",
    "DATE: 12/03/2018",
    "TIME: 14:30",
    "ROC",
    "ROC NO 12345",
    "TEL: 03-1234567",
    "FAX: 03-7654321",
])
def test_addr_terminator_covers_bottom_cut_keywords(text: str) -> None:
    """All bottom-cut keywords from the spec match :data:`_ADDR_TERMINATOR`."""
    assert _ADDR_TERMINATOR.search(text), f"missing terminator match for {text!r}"


@pytest.mark.parametrize("text", [
    "UNIHAKKA INTERNATIONAL SDN BHD",
    "MR D.I.Y (M) SDN BHD",
    "GROCER MART BHD",
    "SOMETHING PTE LTD",
    "ANOTHER CO LTD",
    "GST: 12345678901",
    "GST 12345678901",
    "GST# 999000123",
    "REGNO 123456-A",
    "TAX-ID 123456789012",
])
def test_addr_company_header_covers_pre_anchor_strip(text: str) -> None:
    """Company / tax-ID lines match :data:`_ADDR_COMPANY_HEADER`."""
    assert _ADDR_COMPANY_HEADER.search(text), f"missing company-header match for {text!r}"


# ---------------------------------------------------------------------------
# `_is_addr_boundary` integration — postcode lines never count as boundary.
# ---------------------------------------------------------------------------

def test_postcode_line_is_never_a_boundary() -> None:
    """A 5-digit postcode line is the canonical end of an address."""
    assert _is_addr_boundary("47000 SUNGAI BULOH SELANGOR") is False


def test_company_header_is_a_boundary() -> None:
    assert _is_addr_boundary("UNIHAKKA INTERNATIONAL SDN BHD") is True


def test_doc_no_is_a_boundary() -> None:
    assert _is_addr_boundary("DOC NO 12345") is True


# ---------------------------------------------------------------------------
# `_address_span` — five real failure-receipt fixtures from the miss table.
# Each receipt's leaked tail should be stripped while the postcode line is
# preserved.
# ---------------------------------------------------------------------------

def _bb(n: int) -> list[list[float]]:
    return [[0.0, i * 0.1, 1.0, (i + 1) * 0.1] for i in range(n)]


def test_x00000000_strips_company_header_and_inv_no() -> None:
    """X00000000 — UNIHAKKA INTERNATIONAL SDN BHD ... INV NO ... CASH."""
    texts = [
        "UNIHAKKA INTERNATIONAL SDN BHD",
        "12, JALAN TANJUNG SD 13/2",
        "BANDAR SRI DAMANSARA",
        "52200 KUALA LUMPUR",
        "INV NO 1075214",
        "CASH",
    ]
    out = _address_span(texts, _bb(len(texts)))
    assert "UNIHAKKA" not in out
    assert "BHD" not in out.upper().split()
    assert "INV" not in out.upper().split()
    assert "CASH" not in out.upper().split()
    assert "52200" in out


def test_x00000001_strips_bhd_top_and_receipt_tail() -> None:
    """X00000001 — GROCER MART SDN BHD ... RECEIPT NO ..."""
    texts = [
        "GROCER MART SDN BHD",
        "NO 1, JALAN PJU 7/3",
        "MUTIARA DAMANSARA",
        "47800 PETALING JAYA",
        "RECEIPT NO 9001",
        "DATE: 01/03/2018",
    ]
    out = _address_span(texts, _bb(len(texts)))
    assert "BHD" not in out.upper().split()
    assert "GROCER" not in out.upper()
    assert "RECEIPT" not in out.upper()
    assert "47800" in out


def test_x00000003_strips_international_and_doc_no() -> None:
    """X00000003 — TANCHMAS INTERNATIONAL ... DOC NO ..."""
    texts = [
        "TANCHMAS INTERNATIONAL HOLDINGS",
        "LOT 5, JALAN MAJU",
        "TAMAN MAJU",
        "40000 SHAH ALAM",
        "DOC NO 88421",
        "TAX INVOICE",
    ]
    out = _address_span(texts, _bb(len(texts)))
    assert "INTERNATIONAL" not in out.upper()
    assert "HOLDINGS" not in out.upper()
    assert "DOC" not in out.upper().split()
    assert "TAX" not in out.upper().split()
    assert "40000" in out


def test_x00000008_strips_pte_ltd_and_cashier_tail() -> None:
    """X00000008 — SOMETHING PTE LTD ... CASHIER ... TABLE ..."""
    texts = [
        "ABC TRADING PTE LTD",
        "12 JALAN AMPANG",
        "55000 KUALA LUMPUR",
        "CASHIER: ANNA",
        "TABLE 7",
        "ORDER NO 42",
    ]
    out = _address_span(texts, _bb(len(texts)))
    assert "PTE" not in out.upper().split()
    assert "LTD" not in out.upper().split()
    assert "CASHIER" not in out.upper()
    assert "TABLE" not in out.upper().split()
    assert "ORDER" not in out.upper().split()
    assert "55000" in out


def test_x00000012_strips_gst_id_and_credit_note_tail() -> None:
    """X00000012 — top-of-receipt GST tax-ID + CREDIT NOTE / TEL tail."""
    texts = [
        "AEON CO M BHD",
        "GST: 000123456789",
        "NO 3, PERSIARAN TAMAN",
        "TAMAN PARADIGM",
        "47301 PETALING JAYA",
        "TEL: 03-7882 8888",
        "CREDIT NOTE",
    ]
    out = _address_span(texts, _bb(len(texts)))
    assert "AEON" not in out.upper()
    assert "GST" not in out.upper().split()
    assert "TEL" not in out.upper().split()
    assert "CREDIT" not in out.upper()
    assert "47301" in out


# ---------------------------------------------------------------------------
# `normalize_address_focus` — token-level trailing trim.  Symmetric on
# pred and GT; SROIE GT is clean of these tokens so trim is no-op for gold.
# ---------------------------------------------------------------------------

def test_normalize_address_focus_drops_trailing_company_tokens() -> None:
    """``BHD`` / ``SDN`` / ``INTERNATIONAL`` at the tail are dropped."""
    out = normalize_address_focus("12 JALAN MAJU 47000 SHAH ALAM BHD SDN")
    assert out.endswith("47000 shah alam")


def test_normalize_address_focus_drops_trailing_inv_no_cash() -> None:
    out = normalize_address_focus("12 JALAN MAJU 47000 SHAH ALAM INV NO CASH")
    assert out == "12 jalan maju 47000 shah alam"


def test_normalize_address_focus_drops_short_alpha_fragments() -> None:
    """1-2-char OCR fragments at the tail (``JO``, ``T``, ``#``) drop."""
    out = normalize_address_focus("12 JALAN MAJU 47000 SHAH ALAM JO T")
    assert out == "12 jalan maju 47000 shah alam"


def test_normalize_address_focus_does_not_strip_postcode() -> None:
    """Numeric tokens (postcode) halt the trim — they are the address tail."""
    out = normalize_address_focus("47000 SHAH ALAM")
    assert "47000" in out
    assert out == "47000 shah alam"


def test_normalize_address_focus_clean_gt_is_unchanged_after_trim() -> None:
    """Symmetric application: SROIE GT is clean → trim is a no-op."""
    gt = "NO 12 JALAN MAJU TAMAN MAJU 47000 SHAH ALAM SELANGOR"
    out = normalize_address_focus(gt)
    assert out == gt.casefold()


def test_normalize_address_focus_keeps_internal_company_token() -> None:
    """Trim is anchored to the ENDS — a truly internal ``BHD`` / ``CASH``
    (preceded by a real address token) is preserved so a pathological
    GT containing one of these tokens (rare on SROIE but theoretically
    possible) is not over-trimmed.  PR-ADDR-PREC-2: the leading-token
    trim halts at the first non-boilerplate token, so ``NO 12 BHD …``
    keeps ``bhd`` once ``no 12`` has anchored the head.
    """
    out = normalize_address_focus("NO 12 BHD ROAD 47000 SHAH ALAM")
    # Internal "bhd" survives — the leading trim stops at "no" (kept)
    # and the trailing trim stops at "alam" / postcode.
    assert "bhd" in out.split()


# ---------------------------------------------------------------------------
# `_legacy_address_pick` — boundary filter on the threshold band.
# ---------------------------------------------------------------------------

def test_legacy_address_pick_drops_boundary_lines() -> None:
    """Picks classified as boundary (BHD, INV NO, CASH) are filtered
    BEFORE the spatial-contiguity gate runs.
    """
    torch = pytest.importorskip("torch")
    from models.focus_pipeline import _legacy_address_pick

    texts = [
        "UNIHAKKA INTERNATIONAL SDN BHD",   # 0 — boundary
        "12 JALAN MAJU 5",                   # 1
        "TAMAN MAJU",                        # 2
        "47000 SUNGAI BULOH",                # 3
        "INV NO 1075214 CASH",               # 4 — boundary
    ]
    bboxes = [[0.0, i * 0.10, 1.0, (i + 1) * 0.10] for i in range(5)]
    # Uniform attention → all 5 lines are within the threshold band.
    attn = torch.ones(5)
    picks, value = _legacy_address_pick(attn, texts, bboxes, set(), frac=0.5)
    assert 0 not in picks
    assert 4 not in picks
    assert "UNIHAKKA" not in value
    assert "INV" not in value.upper().split()
    assert "47000" in value
