"""test_address_refiner.py — verify address refinement handles the
observed SROIE miss-table patterns.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: guards the address-side changes in ``pipeline_consensus`` —
    postcode anchor, backward-extend, continuation-line extension, junk
    stripping, and the postcode-preferring candidate scorer — so the
    40+ ``prefix-of-GT`` address misses in the pre-branch miss table
    cannot silently reappear.
"""
from __future__ import annotations

from models.pipeline_consensus import (
    _ADDR_ANCHOR,
    _ADDR_CONTINUATION,
    _POSTCODE_RE,
    _address_span,
    _is_addr_boundary,
    _refine_address,
    _strip_leading_junk,
)


def _bb(y1: float, y2: float) -> list[float]:
    """Simple full-width bbox helper: [x1, y1, x2, y2]."""
    return [0.0, y1, 1.0, y2]


# ----------------------------------------------------------------- anchors ---


def test_anchor_matches_postcode() -> None:
    """5-digit Malaysian postcode is a valid anchor so tail-only OCR
    still produces a non-empty span (then backward-extend recovers the
    street prefix)."""
    assert _ADDR_ANCHOR.search("43200 CHERAS, SELANGOR")


def test_anchor_matches_mall_tokens() -> None:
    assert _ADDR_ANCHOR.search("PARADIGM MALL")
    assert _ADDR_ANCHOR.search("AEON TAMAN MALURI SC")
    assert _ADDR_ANCHOR.search("CITTA MALL, NO 1")


def test_continuation_matches_states_and_cities() -> None:
    assert _ADDR_CONTINUATION.search("SELANGOR DARUL EHSAN")
    assert _ADDR_CONTINUATION.search("KUALA LUMPUR")
    assert _ADDR_CONTINUATION.search("JOHOR BAHRU")


# -------------------------------------------------------- boundary / postcode ---


def test_boundary_does_not_stop_on_postcode_line() -> None:
    """Postcode-bearing tails like ``43200 CHERAS, SELANGOR`` must NOT
    terminate the span — they are the *end* of the address, not outside."""
    assert not _is_addr_boundary("43200 CHERAS, SELANGOR")
    assert not _is_addr_boundary("47500 SUBANG JAYA, SELANGOR.")


def test_boundary_stops_on_tel_and_money() -> None:
    assert _is_addr_boundary("TEL: 03-1234 5678")
    assert _is_addr_boundary("TOTAL RM 43.50")


# --------------------------------------------------------- junk stripping ---


def test_strip_leading_junk_removes_co_no() -> None:
    """Miss #240: ``CO. NO. 37365-A LOT F15, GIANT BANDAR PUTERI`` →
    ``LOT F15, GIANT BANDAR PUTERI``."""
    out = _strip_leading_junk("CO. NO. 37365-A LOT F15, GIANT BANDAR PUTERI")
    assert out.startswith("LOT F15")
    assert "CO." not in out.upper().split()[0]


def test_strip_leading_junk_preserves_clean_address() -> None:
    clean = "LOT 1851-A & 1851-B, JALAN KPB 6, 43300 SERI KEMBANGAN"
    assert _strip_leading_junk(clean) == clean


def test_strip_leading_junk_never_returns_empty() -> None:
    """Safety: if stripping would empty the string, keep the original."""
    only_junk = "CO. NO. 12345"
    assert _strip_leading_junk(only_junk) == only_junk


# ------------------------------------------------------------- span assembly ---


def test_span_includes_postcode_tail() -> None:
    """Miss #136 / #569 / #140 prefix-of-GT: span must append the
    postcode line after the street anchor."""
    texts = [
        "ACME SDN BHD",            # company — skipped (no anchor)
        "NO.2, JALAN TEMENGGUNG 19/9,",    # anchor
        "SEKSYEN 9, BANDAR MAHKOTA CHERAS,",  # continuation
        "43200 CHERAS, SELANGOR",  # postcode tail
        "TEL: 03-1234 5678",       # boundary — stops span
        "TOTAL RM 43.50",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    span = _address_span(texts, bboxes)
    assert "NO.2" in span
    assert "SEKSYEN 9" in span
    assert "43200 CHERAS, SELANGOR" in span
    assert "TEL" not in span


def test_span_backward_extends_to_mall_prefix() -> None:
    """Miss #460 / #303: span should include the mall/brand prefix line
    above the street anchor."""
    texts = [
        "PARADIGM MALL",                         # prefix — backward-extend
        "LOT NO.: 4F-27 & 4FK-38, 4TH FLOOR",    # anchor
        "47301 PETALING JAYA",                   # postcode tail
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    span = _address_span(texts, bboxes)
    assert "PARADIGM MALL" in span
    assert "4TH FLOOR" in span
    assert "47301" in span


def test_span_anchored_on_postcode_backward_extends() -> None:
    """When only the postcode line carries an anchor keyword, the
    backward-extend must recover the street prefix."""
    texts = [
        "NO 290, JALAN AIR PANAS,",
        "SETAPAK,",
        "53200 KUALA LUMPUR.",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    span = _address_span(texts, bboxes)
    assert "JALAN AIR PANAS" in span
    assert "53200 KUALA LUMPUR" in span


def test_span_merges_same_line_regions() -> None:
    """Regions on the same visual line (overlapping y) must join with a
    single space, not be broken across newlines."""
    texts = ["LOT 3,", "JALAN PELABUR 23/1,", "40300 SHAH ALAM"]
    # First two share the same y range → same visual line.
    bboxes = [_bb(0, 1), _bb(0, 1), _bb(2, 3)]
    span = _address_span(texts, bboxes)
    assert "LOT 3, JALAN PELABUR 23/1," in span
    assert "40300 SHAH ALAM" in span


def test_span_empty_when_no_anchor() -> None:
    texts = ["WELCOME", "THANK YOU"]
    bboxes = [_bb(0, 1), _bb(1, 2)]
    assert _address_span(texts, bboxes) == ""


# --------------------------------------------------------- refine scoring ---


def test_refine_prefers_postcode_candidate() -> None:
    """A short prefix-of-GT learned pick must lose to a longer span that
    contains a postcode — even when both are junk-free."""
    texts = [
        "NO.2, JALAN TEMENGGUNG 19/9,",
        "SEKSYEN 9, BANDAR MAHKOTA CHERAS,",
        "43200 CHERAS, SELANGOR",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    learned = "NO.2, JALAN TEMENGGUNG 19/9,"
    out = _refine_address(learned, texts, bboxes, {"company": ""})
    assert _POSTCODE_RE.search(out), f"expected postcode in refined: {out!r}"
    assert "SEKSYEN 9" in out


def test_refine_strips_co_no_junk_from_learned() -> None:
    """Miss #240: learned pick with ``CO. NO. 37365-A`` prefix must be
    de-junked before scoring so the span can still beat it."""
    texts = [
        "CO. NO. 37365-A",
        "LOT F15, GIANT BANDAR PUTERI",
        "JALAN PUTERI 1/1, BDR PUTERI",
        "47100 PUCHONG, SELANGOR",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    learned = "CO. NO. 37365-A LOT F15, GIANT BANDAR PUTERI"
    out = _refine_address(learned, texts, bboxes, {"company": ""})
    assert "CO. NO." not in out.upper()
    assert "47100" in out


# --------------------------- boundary regression (live miss table) ---


def test_span_stops_at_invoice_footer() -> None:
    """Miss #136/#569/#140/#570: the span must NOT run past the postcode
    line into ``INV NO …``/``CASHIER …`` footer even though those lines
    lack money/date/phone markers."""
    texts = [
        "NO.2, JALAN TEMENGGUNG 19/9,",
        "SEKSYEN 9, BANDAR MAHKOTA CHERAS,",
        "43200 CHERAS, SELANGOR",
        "INV NO 1053110",
        "CASHIER THANDAR",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    span = _address_span(texts, bboxes)
    assert "SELANGOR" in span
    assert "INV NO" not in span.upper()
    assert "CASHIER" not in span.upper()


def test_span_backward_rejects_company_header() -> None:
    """Miss #555/#200/#399: ``MR D.I.Y M SDN BHD`` line above a ``LOT …``
    anchor must not be swept into the span by backward-extend."""
    texts = [
        "MR D.I.Y M SDN BHD",
        "LOT 1851-A & 1851-B, JALAN KPB 6,",
        "KAWASAN PERINDUSTRIAN BALAKONG,",
        "43300 SERI KEMBANGAN, SELANGOR",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    span = _address_span(texts, bboxes)
    assert "SDN BHD" not in span.upper()
    assert span.startswith("LOT 1851-A")
    assert "43300" in span


def test_span_backward_rejects_company_with_reg_no() -> None:
    """Miss #525/#195: ``AEON CO M BHD 126926-H`` must not backward-
    extend into the span anchored on ``3RD FLR AEON TAMAN MALURI``."""
    texts = [
        "AEON CO M BHD 126926-H",
        "3RD FLR AEON TAMAN MALURI SC",
        "JLN JEJAKA, TAMAN MALURI",
        "CHERAS, 55100 KUALA LUMPUR",
        "SHOPPING HOURS SUN-THU",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    span = _address_span(texts, bboxes)
    assert "CO M BHD" not in span.upper()
    assert "126926-H" not in span
    assert "3RD FLR" in span
    assert "55100" in span
    assert "SHOPPING HOURS" not in span.upper()


def test_boundary_stops_on_terminator_and_company() -> None:
    assert _is_addr_boundary("INV NO 1053110")
    assert _is_addr_boundary("SIMPLIFIED TAX INVOICE CASH")
    assert _is_addr_boundary("BILL TO SUCI ALAM JAYA")
    assert _is_addr_boundary("MR D.I.Y M SDN BHD")
    assert _is_addr_boundary("POPULAR BOOK CO M SDN BHD")
    # Non-terminators must still be accepted inside an address:
    assert not _is_addr_boundary("43200 CHERAS, SELANGOR")
    assert not _is_addr_boundary("LOT 1851-A & 1851-B, JALAN KPB 6,")
