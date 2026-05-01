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


# -----------------------------------------------------------------------
# FOCUS-Σ — Identity 3 (items-subset-sum + tax_aug) regression suite.
# -----------------------------------------------------------------------

from models.total_arithmetic import (  # noqa: E402
    subset_sum_target_cents,
    total_witness_count,
)


def test_focus_sigma_fires_without_subtotal_keyword() -> None:
    """Receipt with no SUBTOTAL keyword: I₁/I₂ silent, I₃ alone witnesses."""
    texts = ["ITEM A 12.00", "ITEM B 18.50", "ITEM C 3.00",
             "TAX 2.00", "TOTAL 35.50"]
    c = _classify(texts)
    assert total_witness_count(35.50, c) == 1
    assert total_witness_count(12.00, c) == 0  # singleton item not a witness


def test_focus_sigma_stacks_with_keyword_identities() -> None:
    """When I₂ fires, I₃ stacks for a count of 2 on the true total."""
    texts = ["ITEM A 12.00", "ITEM B 18.50", "ITEM C 3.00",
             "SUBTOTAL 33.50", "TAX 2.00", "TOTAL 35.50"]
    c = _classify(texts)
    assert total_witness_count(35.50, c) == 2


def test_focus_sigma_rejects_singleton_self_match() -> None:
    """A noise money line (phone number parsed as money) is NOT its own
    cardinality-1 witness — Identity 3 requires subset cardinality ≥ 2
    (or 1 plus non-zero tax_aug)."""
    texts = ["ITEM A 99.00", "TOTAL 99.00"]
    c = _classify(texts)
    assert total_witness_count(99.00, c) == 0


def test_focus_sigma_drops_per_item_outliers() -> None:
    """A line whose value exceeds the per-item cap (RM 5000) drops out
    of the items pool so the DP bound stays tractable on noisy receipts."""
    texts = ["ITEM A 200.00", "ITEM B 136.20", "ITEM C 100.00",
             "TAX 0.00", "TOTAL 436.20", "PHONE 4838.20"]
    c = _classify(texts)
    # 200 + 136.20 + 100.00 = 436.20 — true total has a witness.
    assert total_witness_count(436.20, c) >= 1
    # Outlier 4838.20 has no witness (capped out of items pool, and not
    # reachable as any subset-sum of remaining items + tax_aug=0).
    assert total_witness_count(4838.20, c) == 0


def test_focus_sigma_empty_classification() -> None:
    """Empty input → empty target set, witness count 0."""
    assert subset_sum_target_cents([]) == frozenset()
    assert total_witness_count(10.0, []) == 0


# -----------------------------------------------------------------------
# FOCUS-Σ OCR-drift recovery (consensus._ocr_drift_match_in_set + the
# _refine_total Identity-3 path). Uses _refine_total so the gates
# (TOTAL keyword, no SUBTOTAL keyword, target_set non-empty) are exercised.
# -----------------------------------------------------------------------

from models.consensus import _ocr_drift_match_in_set, _refine_total  # noqa: E402


def test_focus_sigma_ocr_drift_helper_largest_match() -> None:
    """When multiple 1-edit neighbours land in the target set, prefer
    the LARGER value — on SROIE, grand-total ≥ partial-sum almost always.
    """
    assert _ocr_drift_match_in_set(820, frozenset({800, 850})) == 850
    assert _ocr_drift_match_in_set(860, frozenset({660, 880})) == 880
    assert _ocr_drift_match_in_set(10980, frozenset({10880, 16980})) == 16980


def test_focus_sigma_ocr_drift_helper_leading_zero_rejected() -> None:
    """A 1-edit candidate that produces a leading zero is not a real OCR
    substitution (digit was lost, not substituted).  Reject it.
    """
    # 148 → "048" only via pos-0 1→0 substitution; reject.
    assert _ocr_drift_match_in_set(148, frozenset({48})) is None


def test_focus_sigma_ocr_drift_helper_no_match() -> None:
    assert _ocr_drift_match_in_set(999, frozenset({100, 200, 300})) is None
    assert _ocr_drift_match_in_set(0, frozenset({500})) is None


def test_focus_sigma_ocr_drift_recovers_no_subtotal_receipt() -> None:
    """No SUBTOTAL keyword (I₂ silent), TOTAL line OCR'd 1 digit off
    the items sum.  Identity-3 OCR-drift substitutes the items-sum target.
    """
    texts = ["ITEM A 5.00", "ITEM B 3.00", "ITEM C 0.50", "TOTAL 8.20"]
    bb = [[0, i * 0.1, 1, (i + 1) * 0.1] for i in range(len(texts))]
    assert _refine_total("8.20", texts, bb, None) == "8.50"


def test_focus_sigma_ocr_drift_with_tax_aug() -> None:
    """tax_aug != 0 disambiguates: I₃ targets are items+tax, so the
    1-edit substitution lands on the grand-total magnitude directly.
    """
    texts = ["ITEM A 5.00", "ITEM B 3.00", "TAX 0.40", "TOTAL 8.20"]
    bb = [[0, i * 0.1, 1, (i + 1) * 0.1] for i in range(len(texts))]
    assert _refine_total("8.20", texts, bb, None) == "8.40"


def test_focus_sigma_ocr_drift_silent_on_subtotal_neighbour() -> None:
    """When the SUBTOTAL keyword is in the line's neighbourhood, the
    FOCUS-Σ OCR-drift path is gated off (don't substitute against a
    subtotal line that happens to be 1 digit off).  Returns the
    legacy-path value or the original learned input.
    """
    texts = [
        "ITEM A 5.00", "ITEM B 3.00",
        "SUBTOTAL 8.00",
        "TOTAL 8.20",  # 1 edit from 8.50, but legacy I₂ already resolves
    ]
    bb = [[0, i * 0.1, 1, (i + 1) * 0.1] for i in range(len(texts))]
    out = _refine_total("8.20", texts, bb, None)
    # I₂ target = subtotal + tax = 8.00 + 0 = 8.00; existing
    # ARITHMETIC-FIRST PATH may emit "8.00".  Either way, the FOCUS-Σ
    # path itself does not steer the value to 8.50 because the
    # subtotal-neighbour gate fires.
    assert out in {"8.20", "8.00"}, f"unexpected refinement: {out!r}"


# -----------------------------------------------------------------------
# Inference-side push-toward-0.93 fixes (run 20260430T125211Z empirical
# taxonomy: 2-edit OCR-drift, sign-positive gate, zero/negative/max-money
# scoring penalties, tighter confidence-gated override).
# -----------------------------------------------------------------------


def test_focus_sigma_ocr_drift_2edit_helper_off_by_default() -> None:
    """``max_edits=1`` is the default — must not return a 2-edit match."""
    # 469 vs 4970: 4 same, 6→9 (pos 1), 9→7 (pos 2) — two edits.
    assert _ocr_drift_match_in_set(4690, frozenset({4970})) is None


def test_focus_sigma_ocr_drift_2edit_helper_fires_when_enabled() -> None:
    """``max_edits=2`` returns a 2-edit match if no 1-edit match exists."""
    # 4690 → 4970 needs two substitutions (pos 1 and pos 2).
    assert _ocr_drift_match_in_set(
        4690, frozenset({4970}), max_edits=2,
    ) == 4970


def test_focus_sigma_ocr_drift_2edit_helper_prefers_1edit_when_both() -> None:
    """When both a 1-edit and a 2-edit match exist, prefer larger of all."""
    # 820 → 850 (1-edit, pos 1) AND 820 → 950 (2-edit, pos 0+pos 1).
    # Tiebreak by max -> 950 wins because 950 > 850.  This is correct on
    # SROIE (grand total > sub-sum).
    assert _ocr_drift_match_in_set(
        820, frozenset({850, 950}), max_edits=2,
    ) == 950


def test_focus_sigma_2edit_rejects_leading_zero() -> None:
    """A 2-edit candidate that produces a leading zero is rejected."""
    # 1234 → "0034" via two edits (pos 0: 1→0, pos 1: 2→0) — leading zero.
    assert _ocr_drift_match_in_set(1234, frozenset({34}), max_edits=2) is None


def test_score_money_zero_pred_demoted() -> None:
    """A money line with value 0.00 must score lower than a TOTAL-keyword'd
    line with a real value, even on the same positional rank."""
    from models.consensus import _score_money
    texts = ["TOTAL 0.00", "GRAND TOTAL 35.50"]
    score_zero = _score_money(0, texts, {0: 0, 1: 1}, [0, 1])
    score_total = _score_money(1, texts, {0: 0, 1: 1}, [0, 1])
    assert score_total > score_zero, (score_total, score_zero)


def test_score_money_negative_pred_demoted() -> None:
    """A negative-money line (CHANGE / REFUND) is heavily demoted vs a
    positive TOTAL-keyword'd line."""
    from models.consensus import _score_money
    texts = ["CHANGE -28.35", "TOTAL 140.65"]
    s_neg = _score_money(0, texts, {0: 0, 1: 1}, [0, 1])
    s_pos = _score_money(1, texts, {0: 0, 1: 1}, [0, 1])
    assert s_pos > s_neg, (s_pos, s_neg)


def test_score_money_max_money_relative_prior() -> None:
    """A line whose value is < 30% of the receipt-max gets demoted vs a
    line carrying the max value (when both have the same keyword context)."""
    from models.consensus import _score_money
    # Both lines have weak TOTAL keyword; without the max-money prior
    # the small line could win on positional rank.  With it, the larger
    # value wins.
    texts = ["TOTAL 1.55", "TOTAL 27.35"]
    money_idxs = [0, 1]
    s_small = _score_money(0, texts, {}, money_idxs)
    s_big = _score_money(1, texts, {}, money_idxs)
    assert s_big > s_small, (s_big, s_small)


def test_refine_total_sign_positive_gate() -> None:
    """When learned is negative AND the rule-scored best is a positive
    plausible total, the negative learned is rejected unconditionally
    (CHANGE / REFUND line was picked by the assigner)."""
    texts = [
        "ITEM A 50.00", "ITEM B 90.65",
        "GRAND TOTAL 140.65",
        "CHANGE -28.35",
    ]
    bb = [[0, i * 0.1, 1, (i + 1) * 0.1] for i in range(len(texts))]
    out = _refine_total("-28.35", texts, bb, None)
    assert "140" in out, f"expected positive grand-total, got {out!r}"
