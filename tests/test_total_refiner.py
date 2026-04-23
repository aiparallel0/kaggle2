"""test_total_refiner.py — conservative TOTAL override behaviour.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: guards :func:`models.pipeline_consensus._refine_total` against the
    regression observed on the live checkpoint (total-F1 dropped from
    0.619 → 0.540 when the refiner overrode the learned argmax on any
    ``best_score > 0``).  The margin-based override rule is verified
    here: a well-formed learned money value survives when no competing
    candidate has decisively better keyword evidence, and the classic
    SUBTOTAL-vs-GRAND-TOTAL correction still fires when it does.
"""
from __future__ import annotations

from models.pipeline_consensus import _refine_total


def _bb(y1: float, y2: float) -> list[float]:
    return [0.0, y1, 1.0, y2]


def test_keeps_learned_when_no_strong_competing_evidence() -> None:
    """Both candidates carry ``TOTAL`` — the refiner must NOT flip just
    because one happens to be the last money line on the receipt."""
    texts = [
        "TOTAL RM 43.50",   # learned picked this one
        "SUBTOTAL 43.50",
        "CHANGE 6.50",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    # Learned argmax matches the TOTAL line.
    assert _refine_total("43.50", texts, bboxes, attn_row=None) == "43.50"


def test_overrides_when_learned_is_invalid_money() -> None:
    """Empty / non-money learned pick → take the scored argmax."""
    texts = [
        "ACME SDN BHD",
        "",
        "TOTAL RM 43.50",
        "",
        "THANK YOU",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    assert _refine_total("", texts, bboxes, attn_row=None) == "43.50"


def test_overrides_subtotal_when_grand_total_is_stronger() -> None:
    """Classic SUBTOTAL confusion: learned picked SUBTOTAL but a line
    with GRAND TOTAL is decisively better.  Separator lines ensure the
    ±1-line scorer neighbourhood does not blend the two keywords."""
    texts = [
        "ITEM 1 5.00",
        "ITEM 2 25.00",
        "SUBTOTAL 30.00",
        "TAX 0.00",
        "GRAND TOTAL RM 43.50",
        "THANK YOU",
    ]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    assert _refine_total("30.00", texts, bboxes, attn_row=None) == "43.50"


def test_keeps_learned_when_no_money_regions() -> None:
    """If no line carries a money value, refiner must fall back to the
    learned value unchanged."""
    texts = ["THANK YOU", "COME AGAIN"]
    bboxes = [_bb(y, y + 1) for y in range(len(texts))]
    assert _refine_total("43.50", texts, bboxes, attn_row=None) == "43.50"
