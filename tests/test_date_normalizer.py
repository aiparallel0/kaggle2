"""test_date_normalizer.py — canonical ``DD/MM/YYYY`` across all formats.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: guards :func:`models.pipeline_normalize.normalize_date` against
    the regression observed in the live checkpoint run (date-F1 slid
    from 0.95 → 0.88 because word-form dates and 2-digit years were
    kept as-is while numeric forms were zero-padded — so pred/GT mixed
    conventions leaked token-F1).  The normaliser now canonicalises
    every input family to ``DD/MM/YYYY`` and is symmetric on both
    sides, so format-only mismatches can't recur.
"""
from __future__ import annotations

from models.pipeline_normalize import normalize_date


def test_numeric_dashed_to_slashed() -> None:
    assert normalize_date("19-05-2018") == "19/05/2018"


def test_numeric_dotted_to_slashed() -> None:
    assert normalize_date("19.05.2018") == "19/05/2018"


def test_zero_pads_single_digit_day_and_month() -> None:
    assert normalize_date("1/3/2018") == "01/03/2018"


def test_expands_two_digit_year() -> None:
    """A 2-digit year on a receipt must expand to ``20XX`` so pred and
    GT don't mismatch on the year token."""
    assert normalize_date("15/03/18") == "15/03/2018"
    assert normalize_date("1-3-19") == "01/03/2019"


def test_word_dmy_to_numeric() -> None:
    """``15 MAR 2018`` / ``15-MAR-2018`` / ``15/MAR/2018`` all canonicalise."""
    assert normalize_date("15 MAR 2018") == "15/03/2018"
    assert normalize_date("15-MAR-2018") == "15/03/2018"
    assert normalize_date("15/MAR/2018") == "15/03/2018"
    assert normalize_date("1 MAR 2018") == "01/03/2018"


def test_word_full_month_name() -> None:
    assert normalize_date("15 MARCH 2018") == "15/03/2018"
    assert normalize_date("1 AUGUST 2019") == "01/08/2019"
    assert normalize_date("25 SEPT 2018") == "25/09/2018"


def test_word_mdy_us_style() -> None:
    """US-style ``MMM DD YYYY`` / ``MMM DD, YYYY`` → ``DD/MM/YYYY``."""
    assert normalize_date("MAR 15 2018") == "15/03/2018"
    assert normalize_date("MAR 15, 2018") == "15/03/2018"
    assert normalize_date("AUG-01-2019") == "01/08/2019"


def test_case_insensitive_months() -> None:
    assert normalize_date("15 mar 2018") == "15/03/2018"
    assert normalize_date("15 Mar 2018") == "15/03/2018"


def test_idempotent() -> None:
    """Running the normaliser twice must be a no-op — required for
    symmetric pred/GT normalisation in :func:`_nt`."""
    once = normalize_date("1 MAR 18")
    assert once == "01/03/2018"
    assert normalize_date(once) == once


def test_preserves_already_canonical() -> None:
    assert normalize_date("19/05/2018") == "19/05/2018"


def test_extracts_date_from_noisy_context() -> None:
    """``DATE: 19/05/2018`` in the learned pick → bare canonical date."""
    assert normalize_date("DATE: 19/05/2018") == "19/05/2018"


def test_compact_8digit_ddmmyyyy() -> None:
    """TrOCR sometimes drops separators: ``19052018`` → ``19/05/2018``."""
    assert normalize_date("19052018") == "19/05/2018"


def test_unparseable_returns_collapsed_raw() -> None:
    """No date pattern present → collapsed whitespace, no mangling."""
    assert normalize_date("NOT A DATE") == "NOT A DATE"
    assert normalize_date("") == ""
