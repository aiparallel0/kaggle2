"""Unit tests for the rule-based assignment baseline."""
from __future__ import annotations

from models.rule_based import DATE_RE, MONEY_RE, rule_based_assign


def test_assigns_date_from_regex() -> None:
    texts = ["RECEIPT", "Date 02/03/2024", "TOTAL 10.00"]
    bboxes = [[0, 0.0, 1, 0.1], [0, 0.3, 1, 0.4], [0, 0.9, 1, 1.0]]
    out = rule_based_assign(texts, bboxes)
    assert out["date"] == "02/03/2024"


def test_total_is_bottom_money() -> None:
    texts = ["SUBTOTAL 5.00", "TOTAL 10.00", "FOOTER"]
    bboxes = [[0, 0.3, 1, 0.4], [0, 0.8, 1, 0.9], [0, 0.95, 1, 1.0]]
    out = rule_based_assign(texts, bboxes)
    assert out["total"] == "10.00"


def test_company_is_topmost_unused_region() -> None:
    texts = ["ACME SHOP", "02/03/2024", "123 Main St", "TOTAL 10.00"]
    bboxes = [
        [0, 0.0, 1, 0.1], [0, 0.2, 1, 0.3],
        [0, 0.4, 1, 0.5], [0, 0.9, 1, 1.0],
    ]
    out = rule_based_assign(texts, bboxes)
    assert out["company"] == "ACME SHOP"


def test_address_concatenates_multiple_lines_and_skips_money() -> None:
    texts = ["ACME", "123 Main St", "Apt 4B", "Springfield", "TOTAL 10.00"]
    bboxes = [
        [0, 0.0, 1, 0.1], [0, 0.2, 1, 0.3], [0, 0.4, 1, 0.5],
        [0, 0.6, 1, 0.7], [0, 0.9, 1, 1.0],
    ]
    out = rule_based_assign(texts, bboxes)
    assert "Main St" in out["address"]
    assert "Springfield" in out["address"]
    assert "10.00" not in out["address"]


def test_empty_input_returns_empty_dict() -> None:
    assert rule_based_assign([], []) == {}


def test_money_regex_handles_thousands_separator() -> None:
    texts = ["HEADER", "GRAND TOTAL 1,234.56"]
    bboxes = [[0, 0.0, 1, 0.1], [0, 0.9, 1, 1.0]]
    out = rule_based_assign(texts, bboxes)
    assert out["total"] == "1,234.56"


def test_total_prefers_grand_total_over_later_change_line() -> None:
    """On SROIE the last money figure is usually CHANGE / CASH TENDERED —
    not the GRAND TOTAL. The heuristic must prefer the labelled grand
    total even though it is above a later money figure."""
    texts = [
        "COMPANY",
        "SUBTOTAL 10.00",
        "GRAND TOTAL 10.00",
        "CASH TENDERED 20.00",
        "CHANGE 10.00",
    ]
    bboxes = [
        [0, 0.0, 1, 0.1], [0, 0.4, 1, 0.5],
        [0, 0.55, 1, 0.65], [0, 0.7, 1, 0.8], [0, 0.85, 1, 0.95],
    ]
    out = rule_based_assign(texts, bboxes)
    assert out["total"] == "10.00"


def test_total_skips_change_and_rounding_lines() -> None:
    """Negative keywords (CHANGE / ROUNDING) disqualify the region even
    though it's the bottom-most money figure."""
    texts = [
        "ACME",
        "TOTAL 45.60",
        "ROUNDING 0.01",
        "CHANGE 4.40",
    ]
    bboxes = [
        [0, 0.0, 1, 0.1], [0, 0.6, 1, 0.7],
        [0, 0.8, 1, 0.85], [0, 0.9, 1, 0.95],
    ]
    out = rule_based_assign(texts, bboxes)
    assert out["total"] == "45.60"


def test_total_strips_currency_prefix() -> None:
    texts = ["HEADER", "TOTAL RM 12.30"]
    bboxes = [[0, 0.0, 1, 0.1], [0, 0.9, 1, 1.0]]
    out = rule_based_assign(texts, bboxes)
    assert out["total"] == "12.30"


def test_date_accepts_word_month() -> None:
    """SROIE receipts use 01-AUG-2019 and August 1 2019 style formats."""
    assert DATE_RE.search("01-AUG-2019") is not None
    assert DATE_RE.search("1 August 2019") is not None
    assert DATE_RE.search("August 1, 2019") is not None
    assert DATE_RE.search("AUG 1 2019") is not None


def test_date_still_accepts_numeric_formats() -> None:
    assert DATE_RE.search("02/03/2024") is not None
    assert DATE_RE.search("2-3-24") is not None
    assert DATE_RE.search("2024.03.02") is not None


def test_money_handles_currency_prefix() -> None:
    assert MONEY_RE.search("RM12.30") is not None
    assert MONEY_RE.search("$12.30") is not None
    assert MONEY_RE.search("TOTAL 1,234.56") is not None
    # No decimal → not money.
    assert MONEY_RE.search("1234") is None


def test_company_skips_tax_invoice_header_line() -> None:
    """Top region 'TAX INVOICE' must not be picked as company — the
    actual company name is the next line down."""
    texts = ["TAX INVOICE", "ACME CORP SDN BHD", "123 MAIN ST", "TOTAL 10.00"]
    bboxes = [
        [0, 0.0, 1, 0.05], [0, 0.1, 1, 0.15],
        [0, 0.2, 1, 0.25], [0, 0.9, 1, 1.0],
    ]
    out = rule_based_assign(texts, bboxes)
    assert out["company"] == "ACME CORP SDN BHD"


def test_address_excludes_phone_and_tax_id_lines() -> None:
    """Phone numbers, GST IDs etc. sit in the address block spatially
    but must not end up in the concatenated address value."""
    texts = [
        "ACME CORP",
        "NO. 10 JALAN SERI",
        "50450 KUALA LUMPUR",
        "TEL: 03-12345678",
        "GST REG NO: 001234567890",
        "01/01/2024",
        "TOTAL 10.00",
    ]
    bboxes = [
        [0, 0.0, 1, 0.05], [0, 0.1, 1, 0.15],
        [0, 0.2, 1, 0.25], [0, 0.3, 1, 0.35],
        [0, 0.4, 1, 0.45], [0, 0.6, 1, 0.65], [0, 0.9, 1, 1.0],
    ]
    out = rule_based_assign(texts, bboxes)
    assert "JALAN" in out["address"]
    assert "KUALA LUMPUR" in out["address"]
    assert "TEL" not in out["address"]
    assert "GST" not in out["address"]
