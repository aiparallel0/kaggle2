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
# Internal punctuation to collapse to a single space inside company/address
# strings before token-F1 so pred ``"ACME SDN. BHD."`` and GT ``"ACME SDN
# BHD"`` reduce to the same whitespace-token set.  Numeric separators
# (``/`` in dates, ``.`` in decimals) are handled by the field-specific
# normalisers and are NOT collapsed here.
_TEXT_INTERNAL_PUNCT_RE = re.compile(r"[,;:()\[\]\"'`]+|\.(?=\s|$)|(?<=\s)\.")


def _collapse_ws(s: str) -> str:
    return _MULTI_WS_RE.sub(" ", s).strip()


def _strip_edge_punct(s: str) -> str:
    return _LEADING_PUNCT_RE.sub("", _TRAILING_PUNCT_RE.sub("", s)).strip()


def _strip_text_punct(s: str) -> str:
    """Collapse SROIE-GT-inconsistent punctuation inside company/address."""
    return _TEXT_INTERNAL_PUNCT_RE.sub(" ", s)


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
    r"""Collapse whitespace, strip edge + internal punctuation, repair OCR alpha.

    The SROIE GT company values are upper-case single-line strings that
    carry inconsistent trailing ``"SDN BHD."`` vs ``"SDN BHD"`` — each
    such mismatch costs a full token-F1 point.  :func:`_strip_text_punct`
    collapses ``,;:()[]."'\``` and orphan periods inside the value so
    both sides of the comparison end up with the same token set.  Case
    is preserved; token-F1 is case-insensitive already.  Digit-into-alpha
    OCR confusions are first repaired via :func:`repair_company_ocr`.
    """
    t = _strip_text_punct(repair_company_ocr(value))
    return _strip_edge_punct(_collapse_ws(t))


def normalize_address(value: str) -> str:
    """Collapse whitespace, strip edge + internal punctuation, repair postcodes.

    Address GT on SROIE mixes comma and no-comma conventions
    (``"NO 12, BLOCK 3, 50100 KL"`` vs ``"NO 12 BLOCK 3 50100 KL"``);
    stripping these symmetrically via :func:`_strip_text_punct` avoids
    the per-comma F1 penalty.  :func:`repair_postcode_ocr` fixes digit
    confusions inside 5-digit postcode-shaped runs only — Malaysian
    SROIE addresses are postcode-majority — so token-F1 is not
    penalised for a single OCR letter/digit swap in the postcode.
    """
    t = _strip_text_punct(repair_postcode_ocr(value))
    return _strip_edge_punct(_collapse_ws(t))


def normalize_total_value(value: str) -> str:
    """Delegate to ``normalize_total`` — shared with the DONUT eval path."""
    return normalize_total(value)
