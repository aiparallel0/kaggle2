"""Per-field output normalizers — the fourth step of the ``date`` recipe.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: date/company/address/total string normalisation applied after the
    learned+rule consensus picks a value.  Factored out of
    ``pipeline_consensus`` so each field's normaliser stays small and
    independently unit-reviewable.  TOTAL normalisation is delegated to
    ``models.donut_eval.normalize_total`` so pred/GT comparison stays
    symmetric with the DONUT arm.
"""
from __future__ import annotations

import re

from models.donut_eval import normalize_total
from models.pipeline_corrections import (
    repair_company_ocr,
    repair_date_ocr,
    repair_postcode_ocr,
)
from models.rule_regex import _DATE_RE

__all__ = [
    "normalize_address",
    "normalize_company",
    "normalize_date",
    "normalize_total_value",
]

_MULTI_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s,;:.\-_]+$")
_LEADING_PUNCT_RE = re.compile(r"^[\s,;:.\-_]+")
_NUM_DATE_RE = re.compile(r"^(\d{1,4})[/\-\.](\d{1,2})[/\-\.](\d{1,4})$")


def _collapse_ws(s: str) -> str:
    return _MULTI_WS_RE.sub(" ", s).strip()


def _strip_edge_punct(s: str) -> str:
    return _LEADING_PUNCT_RE.sub("", _TRAILING_PUNCT_RE.sub("", s)).strip()


def normalize_date(value: str) -> str:
    """Extract a DATE_RE match and canonicalise numeric separators to ``/``.

    Applies :func:`repair_date_ocr` first so compact TrOCR outputs like
    ``"12032026"`` become ``"12/03/2026"`` and common digit confusions
    (``"l2/O3/2O26"`` → ``"12/03/2026"``) are recovered before the regex
    extractor runs.  Numeric three-part dates (``15-03-2018``,
    ``15.03.2018``) all become ``15/03/2018`` so pred/GT compare equal
    under whitespace-token F1.  Word dates (``15 MAR 2018``) are kept
    as-is.  Falls back to the raw value on no regex hit.
    """
    repaired = repair_date_ocr(value)
    m = _DATE_RE.search(repaired)
    if m is None:
        return _collapse_ws(repaired)
    raw = m.group(0)
    num = _NUM_DATE_RE.match(raw.strip())
    if num is None:
        return _collapse_ws(raw)
    return f"{num.group(1)}/{num.group(2)}/{num.group(3)}"


def normalize_company(value: str) -> str:
    """Collapse whitespace, strip edge punctuation, and repair OCR alpha→digit.

    The SROIE GT company values are upper-case single-line strings, so we
    do not case-fold here — the metric already lower-cases both sides
    before token F1.  :func:`repair_company_ocr` fixes pure-alpha tokens
    (``"SDN 8HD"`` → ``"SDN BHD"``) while leaving address numerals
    (``"BLOCK 3"``, ``"LOT 8A"``) untouched.
    """
    return _strip_edge_punct(_collapse_ws(repair_company_ocr(value)))


def normalize_address(value: str) -> str:
    """Collapse whitespace, strip edge punctuation, repair OCR 5-digit postcodes.

    :func:`repair_postcode_ocr` corrects digit confusions (``O``→``0``,
    ``l``→``1``) inside 5-digit postcode-shaped runs only — Malaysian
    SROIE addresses are postcode-majority — so token F1 is not
    penalised for a single OCR letter/digit swap in the postcode.
    """
    return _strip_edge_punct(_collapse_ws(repair_postcode_ocr(value)))


def normalize_total_value(value: str) -> str:
    """Delegate to ``normalize_total`` — shared with the DONUT eval path."""
    return normalize_total(value)
