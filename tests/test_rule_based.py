"""Unit tests for the rule-based assignment baseline."""
from __future__ import annotations

from models.rule_based import rule_based_assign


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
