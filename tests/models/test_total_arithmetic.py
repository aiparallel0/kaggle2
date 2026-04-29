"""Unit tests for the FOCUS-T arithmetic-consensus solver.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: lock the four arithmetic identities of
    :func:`models.total_arithmetic.total_arithmetic_consensus` and the
    six concrete failure modes from the live miss table (cash-tendered
    confusion, GST-line confusion, OCR digit corruption, subtotal-vs-
    grand-total confusion, negative refund totals, and abstention on
    under-determined receipts).
"""
from __future__ import annotations

from models.total_arithmetic import (
    _classify,
    _identity_cash_change,
    _identity_sub_tax,
    _parse_money,
    total_arithmetic_consensus,
)
from models.total_post import extract_total_value


def test_parse_money_rightmost() -> None:
    """Picks the rightmost money on the line."""
    assert _parse_money("ITEM 5.00 SUB 12.50") == 12.50
    assert _parse_money("RM 1,234.56") == 1234.56
    assert _parse_money("just text") is None


def test_parse_money_negative() -> None:
    """Refund / credit-note lines preserve the leading minus."""
    assert _parse_money("REFUND -6.42") == -6.42


def test_classify_precedence_subtotal_over_total() -> None:
    """A SUB-TOTAL line is never mislabelled as a grand total."""
    out = _classify(["SUB-TOTAL 12.50", "TOTAL 13.25"])
    labels = {lab for _i, lab, _v in out}
    assert "subtotal" in labels and "total" in labels


def test_classify_cash_above_change() -> None:
    """Cash precedence is firm; KEMBALIAN never re-routes to cash."""
    out = _classify(["TUNAI 100.00", "KEMBALIAN 51.10"])
    labels = [lab for _i, lab, _v in out]
    assert labels == ["cash", "change"]


def test_cash_change_identity_recovers_X00000305() -> None:
    """Mode A: cash − change = total when both unique-valued."""
    texts = ["ITEMS", "5.50", "TUNAI 100.00", "KEMBALIAN 94.50"]
    pick = total_arithmetic_consensus(texts, set())
    assert pick is not None and pick[1] == "5.50"


def test_sub_tax_identity_recovers_X00000045() -> None:
    """Mode B: subtotal + tax = total even when GST line itself is the
    bottom-up regex pick."""
    texts = ["SUB-TOTAL 6.98", "GST 6% 0.42", "TOTAL 7.40"]
    pick = total_arithmetic_consensus(texts, set())
    assert pick is not None and pick[1] == "7.40"


def test_sub_tax_with_service_recovers_X00000022() -> None:
    """Mode D: subtotal + service = total."""
    texts = ["SUBTOTAL 371.00", "SERVICE CHARGE 41.90", "GRAND TOTAL 412.90"]
    pick = total_arithmetic_consensus(texts, set())
    assert pick is not None and pick[1] == "412.90"


def test_synthesised_value_when_total_line_corrupted() -> None:
    """Mode C: OCR mangled the total line — synthesise from arithmetic.

    Caller distinguishes synthesis (idx == -1) from line-anchored picks
    (idx >= 0) so the residual money lines remain available to other
    fields' fallbacks.
    """
    texts = ["SUBTOTAL 800.00", "TAX 48.00", "TOTAL 48.00"]  # 848 → 48
    pick = total_arithmetic_consensus(texts, set())
    assert pick is not None
    idx, value = pick
    assert value == "848.00"
    assert idx == -1  # no line carries 848.00 verbatim


def test_two_of_two_consensus_overrides_singletons() -> None:
    """Two identities agreeing wins regardless of any single-witness
    fallback."""
    texts = [
        "SUBTOTAL 45.00", "TAX 3.90", "TOTAL 48.90",
        "TUNAI 100.00", "KEMBALIAN 51.10",
    ]
    pick = total_arithmetic_consensus(texts, set())
    assert pick is not None and pick[1] == "48.90"


def test_abstain_on_under_determined() -> None:
    """No subtotal, no cash/change pair → return ``None`` so the caller
    falls through to the legacy chain rather than guessing."""
    texts = ["ITEM A 5.00", "ITEM B 7.00", "TOTAL 12.00"]
    assert total_arithmetic_consensus(texts, set()) is None


def test_abstain_when_cash_change_negative() -> None:
    """``cash < change`` is a parse error / OCR confusion — refuse to
    return a negative consensus value masquerading as a total."""
    texts = ["TUNAI 50.00", "KEMBALIAN 100.00"]
    assert total_arithmetic_consensus(texts, set()) is None


def test_extract_total_value_preserves_negative() -> None:
    """Mode E: extract_total_value preserves the leading minus on a
    refund / credit-note line."""
    assert extract_total_value("TOTAL -6.42") == "-6.42"
    assert extract_total_value("RM -5.09") == "-5.09"


def test_extract_total_value_positive_unchanged() -> None:
    """The negative-sign fix must not regress the canonical positive
    case the existing tests cover."""
    assert extract_total_value("TOTAL RM 115.00") == "115.00"
    assert extract_total_value("12.34") == "12.34"


def test_consensus_skips_used_lines() -> None:
    """Lines already consumed by another field do not vote in the
    arithmetic identities (otherwise the cash line a previous field
    grabbed could still satisfy ``cash − change``)."""
    texts = ["TUNAI 100.00", "KEMBALIAN 51.10"]
    used = {0}  # cash line consumed elsewhere
    assert total_arithmetic_consensus(texts, used) is None


def test_identity_helpers_return_none_when_inputs_missing() -> None:
    """Defensive: identity helpers return None on degenerate classifier
    output (no subtotal, no cash/change) so the consensus loop never
    sees a spurious zero candidate."""
    assert _identity_cash_change([]) is None
    assert _identity_sub_tax([]) is None
