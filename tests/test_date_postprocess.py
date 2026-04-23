"""test_date_postprocess.py — SROIE date sanity filter (Fix C).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: guards :mod:`models.date_postprocess` — the post-assignment
    sanity filter that rejects OCR-glitch dates (year outside
    2014–2019, malformed day/month) and falls back to the first
    plausible date scanned from the OCR line set.  Backs the
    problem-statement claim that receipts 215, 403, 106, 021 flip
    from wrong-year picks to a correct rule-based alternative.
"""
from __future__ import annotations

from models.date_postprocess import fallback_from_ocr_lines, is_plausible


def test_plausible_in_window() -> None:
    assert is_plausible("19/05/2018")
    assert is_plausible("01/01/2014")
    assert is_plausible("31/12/2019")


def test_rejects_year_before_window() -> None:
    assert not is_plausible("04/12/2012")  # receipt 183
    assert not is_plausible("08/08/2008")  # receipt 021


def test_rejects_year_after_window() -> None:
    assert not is_plausible("18/06/2020")  # receipt 538


def test_rejects_malformed() -> None:
    assert not is_plausible("1851/03/3")   # receipt 403
    assert not is_plausible("01/00/2")     # receipt 106
    assert not is_plausible("")
    assert not is_plausible("not a date")


def test_rejects_out_of_range_day_or_month() -> None:
    assert not is_plausible("32/05/2018")
    assert not is_plausible("15/13/2018")
    assert not is_plausible("00/05/2018")


def test_fallback_returns_first_plausible() -> None:
    lines = [
        "RECEIPT #1851/03/3",            # implausible, skipped
        "DATE: 15/03/2018",              # plausible, returned
        "OTHER 19/05/2018",
    ]
    assert fallback_from_ocr_lines(lines) == "15/03/2018"


def test_fallback_expands_two_digit_year() -> None:
    assert fallback_from_ocr_lines(["PAID ON 5/3/18"]) == "05/03/2018"


def test_fallback_returns_none_when_no_plausible() -> None:
    assert fallback_from_ocr_lines(["NO DATE HERE", "08/08/2008"]) is None


def test_fallback_skips_out_of_range_years() -> None:
    lines = ["TOTALLING 04/12/2012", "CORRECT 22/04/2016"]
    assert fallback_from_ocr_lines(lines) == "22/04/2016"
