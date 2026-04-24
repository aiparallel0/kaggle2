"""Tests for strategies B (hard-negatives) and C (rule-based KD teacher).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: verify that :mod:`models.assigner_teacher` extracts the right
    distractor region sets (SUBTOTAL/CASH/CHANGE/TAX for total,
    phone/header for address/company) and that the rule-based teacher
    distribution puts mass on the GRAND-TOTAL line of a synthetic
    SUBTOTAL/TOTAL receipt.
"""
from __future__ import annotations

from models.assigner_teacher import (
    hard_negatives,
    teacher_distribution,
)


def test_hard_negatives_total_includes_subtotal_tax_cash() -> None:
    texts = [
        "ACME STORE",           # 0
        "ITEM 10.00",           # 1 — extra money line → negative for total
        "SUBTOTAL 30.00",       # 2 — classic confuser
        "TAX 3.00",             # 3
        "CASH TENDERED 50.00",  # 4
        "CHANGE 6.50",          # 5
        "GRAND TOTAL 43.50",    # 6 — GT positive
    ]
    positives = {0: [6]}
    field_to_idx = {"total": 0}
    negs = hard_negatives(texts, positives, field_to_idx)
    assert 0 in negs
    neg_set = set(negs[0])
    # Must flag the canonical confusers.
    for expected in (2, 3, 4, 5):
        assert expected in neg_set, f"missing hard-neg for line {expected}"
    # Must NOT flag the GT line.
    assert 6 not in neg_set


def test_hard_negatives_address_excludes_header_and_phone() -> None:
    texts = [
        "TAX INVOICE",                 # 0 — header junk → negative
        "ACME SDN BHD",                # 1
        "NO. 12 JALAN ABC",            # 2 — positive
        "43200 CHERAS, SELANGOR",      # 3 — positive
        "TEL: 03-1234-5678",           # 4 — phone → negative
        "GST NO 12345",                # 5 — tax-id → negative
    ]
    positives = {0: [2, 3]}
    field_to_idx = {"address": 0}
    negs = hard_negatives(texts, positives, field_to_idx)
    assert set(negs.get(0, [])).issuperset({0, 4, 5})
    # Positives are not in the negative set.
    assert 2 not in negs.get(0, [])
    assert 3 not in negs.get(0, [])


def test_hard_negatives_skips_unknown_fields() -> None:
    """Only total/address/company get distractor sets; date is ceiling."""
    texts = ["01/01/2020", "SUBTOTAL 30.00"]
    positives = {0: [0]}
    field_to_idx = {"date": 0}
    assert hard_negatives(texts, positives, field_to_idx) == {}


def test_teacher_distribution_total_peaks_on_grand_total() -> None:
    """The rule-based teacher must place its probability mass on GRAND
    TOTAL rather than SUBTOTAL — this is strategy C's whole purpose."""
    # ``_score_money`` uses a ±1-line keyword window, so we space the
    # lines so the positive keyword only neighbours the GRAND TOTAL.
    texts = [
        "ITEM A 10.00",     # 0 — money line, no keyword context
        "ITEM B 20.00",     # 1 — money line, no keyword context
        "SUBTOTAL 30.00",   # 2 — classic confuser
        "GRAND TOTAL 43.50",  # 3 — the answer
        "THANK YOU",        # 4 — breaks the CHANGE neighbourhood
    ]
    probs = teacher_distribution(texts, {"total": 0})
    p = probs.get(0)
    assert p is not None
    # Softmax mass peaks on GRAND TOTAL (idx 3), strictly above SUBTOTAL (2).
    assert p[3] == max(p)
    assert p[3] > p[2]
    assert abs(sum(p) - 1.0) < 1e-6


def test_teacher_distribution_address_uniform_over_valid_lines() -> None:
    """Address teacher drops phone/tax-id/header/money/date lines and
    spreads mass uniformly over the remaining candidates."""
    texts = [
        "TAX INVOICE",           # 0 — header → excluded
        "ACME SDN BHD",          # 1 — candidate
        "NO. 12 JALAN ABC",      # 2 — candidate
        "TEL 03-1234-5678",      # 3 — phone → excluded
        "TOTAL 43.50",           # 4 — money → excluded
        "01/01/2020",            # 5 — date → excluded
    ]
    probs = teacher_distribution(texts, {"address": 0})
    p = probs.get(0)
    assert p is not None
    assert abs(sum(p) - 1.0) < 1e-6
    # Only indices 1, 2 carry mass.
    assert p[1] > 0 and p[2] > 0
    assert p[0] == p[3] == p[4] == p[5] == 0.0


def test_teacher_distribution_empty_on_no_money() -> None:
    """No money on the receipt → no teacher signal for ``total``."""
    texts = ["ACME STORE", "THANK YOU"]
    probs = teacher_distribution(texts, {"total": 0})
    assert 0 not in probs
