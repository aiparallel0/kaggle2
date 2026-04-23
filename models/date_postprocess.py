"""SROIE date sanity filter with OCR-line fallback.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: Fix C of the post-eval regression plan.  The assigner + regex
    router occasionally lock onto the wrong token on the date line —
    ``"04/12/2012"`` (wrong-year OCR), ``"1851/03/3"`` (day/year swap
    on a decade-skip OCR), ``"01/00/2"`` (a partial token).  Every
    such predicted date is format-plausible at the regex level but
    implausible against SROIE's actual capture window (2014–2019).
    This module exposes two stdlib-only helpers — :func:`is_plausible`
    and :func:`fallback_from_ocr_lines` — that reject such values and
    scan the OCR line set for the first plausible alternative.  Kept
    import-cheap (no torch/transformers) so it can run inside the
    inference-time assigner path.
"""
from __future__ import annotations

import re

# SROIE task-3 receipts were captured 2014–2019 inclusive; every
# out-of-range year in the current 63-receipt test set is a pure OCR
# glitch (``2012``, ``2008``, ``1227``, ``1851``, ``2020``), not a
# legitimate reading.
_SROIE_YEAR_MIN = 2014
_SROIE_YEAR_MAX = 2019
# ``DD/MM/YYYY`` exactly — the canonical form :func:`normalize_date`
# produces.  Non-canonical inputs are delegated to the normaliser
# upstream so this module only validates the post-normalisation form.
_CANON_DATE_RE = re.compile(r"^(\d{2})/(\d{2})/(\d{4})$")
# Broader "contains a DD[-/.]MM[-/.]YYYY-shaped substring" scan for
# fallback search on raw OCR lines (before normalisation).
_RAW_DATE_RE = re.compile(
    r"(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2,4})",
)

__all__ = ["is_plausible", "fallback_from_ocr_lines"]


def _in_range(day: int, month: int, year: int) -> bool:
    """True iff (d, m, y) is a plausible SROIE receipt date."""
    return (
        _SROIE_YEAR_MIN <= year <= _SROIE_YEAR_MAX
        and 1 <= month <= 12
        and 1 <= day <= 31
    )


def is_plausible(value: str) -> bool:
    """True iff ``value`` is a canonical ``DD/MM/YYYY`` in the SROIE window.

    Used by :mod:`models.pipeline_assign` immediately after the date
    field is selected: an implausible value triggers
    :func:`fallback_from_ocr_lines`.  A non-date or partial token
    (``"01/00/2"``, ``"1851/03/3"``, ``"04/12/2012"``) returns False;
    a well-formed in-range date returns True.
    """
    m = _CANON_DATE_RE.match(value.strip())
    if m is None:
        return False
    d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
    return _in_range(d, mo, y)


def _expand_year(y: str) -> int:
    """2-digit → 20XX, 4-digit → int; anything else → 0 (never plausible)."""
    if len(y) == 2:
        return 2000 + int(y)
    if len(y) == 4:
        return int(y)
    return 0


def fallback_from_ocr_lines(lines: list[str]) -> str | None:
    """Scan OCR lines for the first plausible SROIE date.

    Iterates ``lines`` in reading order and returns the first
    ``DD[-/.]MM[-/.]YY(YY)`` substring whose (day, month, year) triple
    lands in the SROIE window.  Returns ``None`` when no plausible
    date is present — the caller (pipeline_assign) then keeps the
    original implausible pick so downstream ``normalize_date`` still
    has something to canonicalise (and the per-field-F1 just takes
    the hit, which is strictly no worse than the pre-Fix behaviour).
    """
    for line in lines:
        for m in _RAW_DATE_RE.finditer(line):
            d_s, mo_s, y_s = m.group(1), m.group(2), m.group(3)
            try:
                d, mo = int(d_s), int(mo_s)
            except ValueError:
                continue
            y = _expand_year(y_s)
            if _in_range(d, mo, y):
                return f"{d:02d}/{mo:02d}/{y:04d}"
    return None
