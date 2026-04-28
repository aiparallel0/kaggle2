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

from models.corrections import (
    repair_company_ocr,
    repair_date_ocr,
    repair_postcode_ocr,
)
from models.donut_eval import normalize_total
from models.rule_regex import _DATE_RE

__all__ = [
    "normalize_address",
    "normalize_company",
    "normalize_date",
    "normalize_total_value",
    "strip_company_registration",
]

_MULTI_WS_RE = re.compile(r"\s+")
_TRAILING_PUNCT_RE = re.compile(r"[\s,;:.\-_]+$")
_LEADING_PUNCT_RE = re.compile(r"^[\s,;:.\-_]+")
_NUM_DATE_RE = re.compile(r"^(\d{1,4})[/\-\.](\d{1,2})[/\-\.](\d{1,4})$")
# Word-form date normalisation (``1 MAR 2018``, ``1-MAR-2018``,
# ``1/MAR/2018``, ``MAR 1, 2018``, ``AUG 01 2019``) → canonical numeric
# ``DD/MM/YYYY``.  GT on SROIE mixes word and numeric formats for the
# same receipt date; keeping both as-is leaks a full token-F1 point per
# mismatch.  The month alternations are kept in sync with
# :data:`models.rule_regex._MONTHS` so the refiner extracts and the
# normaliser canonicalises the same set.
_MONTH_MAP: dict[str, str] = {
    "jan": "01", "january": "01", "feb": "02", "february": "02",
    "mar": "03", "march": "03", "apr": "04", "april": "04",
    "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10", "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}
_MONTH_ALT = (r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
              r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
              r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?")
_WORD_DATE_DMY_RE = re.compile(
    rf"^(\d{{1,2}})[\s/\-\.]+({_MONTH_ALT})[\s/\-\.]+(\d{{2,4}})$",
    re.IGNORECASE,
)
_WORD_DATE_MDY_RE = re.compile(
    rf"^({_MONTH_ALT})[\s/\-\.]+(\d{{1,2}})[,\s/\-\.]+(\d{{2,4}})$",
    re.IGNORECASE,
)
# Internal punctuation to collapse to a single space inside company/address
# strings before token-F1 so pred ``"ACME SDN. BHD."`` and GT ``"ACME SDN
# BHD"`` reduce to the same whitespace-token set.  Numeric separators
# (``/`` in dates, ``.`` in decimals) are handled by the field-specific
# normalisers and are NOT collapsed here.
_TEXT_INTERNAL_PUNCT_RE = re.compile(r"[,;:()\[\]\"'`]+|\.(?=\s|$)|(?<=\s)\.")
# Fix D — trailing legal-registration suffix that appears on SROIE
# ``company`` OCR lines (``"AEON CO M BHD 126926-H"``) but is NOT in
# the SROIE GT (``"AEON CO M BHD"``).  The pattern matches any of:
#   * a 5–9-digit registration number, optionally followed by ``-X``
#     or `` X`` where ``X`` is a single A–Z check letter
#     (``126926-H``, ``1248446 V``, ``862725-U``);
#   * a bare parenthesised check letter ``(M)`` / ``(X)``;
#   * a 1–2-letter orphan trailing token left over when the registration
#     number was already stripped by an earlier OCR pass.
# Anchored to end-of-string and applied up to twice so a two-suffix
# tail (``SDN BHD 139386 X (M)``) is fully stripped.
_COMPANY_REG_SUFFIX = re.compile(
    r"\s+(?:\d{5,9}[\s\-]?[A-Z]?|\([A-Z]\)|[A-Z]{1,2})\s*$",
)


def _collapse_ws(s: str) -> str:
    return _MULTI_WS_RE.sub(" ", s).strip()


def _strip_edge_punct(s: str) -> str:
    return _LEADING_PUNCT_RE.sub("", _TRAILING_PUNCT_RE.sub("", s)).strip()


def _strip_text_punct(s: str) -> str:
    """Collapse SROIE-GT-inconsistent punctuation inside company/address."""
    return _TEXT_INTERNAL_PUNCT_RE.sub(" ", s)


def _expand_year(y: str) -> str:
    """2-digit receipt years → 20XX; keep 4-digit as-is."""
    if len(y) == 2:
        return f"20{y}"
    if len(y) == 3:  # extremely rare OCR 3-digit year; left-pad with 2
        return f"2{y}"
    return y


def normalize_date(value: str) -> str:
    """Canonicalise a date to ``DD/MM/YYYY`` regardless of input format.

    Pipeline:

    1. :func:`repair_date_ocr` first so compact TrOCR outputs
       (``"12032026"``) and common digit confusions (``"l2/O3/2O26"``)
       are recovered into a numeric form.
    2. Extract the first date-shaped substring via
       :data:`models.rule_regex._DATE_RE`.
    3. Canonicalise four families:

       * ``DD[-/.]MM[-/.]YYYY`` (numeric three-part) → ``DD/MM/YYYY``,
         zero-padded, year expanded from 2-digit → ``20XX``.
       * ``D MMM YYYY`` / ``D-MMM-YYYY`` / ``D/MMM/YYYY`` (word-DMY) →
         ``DD/MM/YYYY``.
       * ``MMM DD YYYY`` / ``MMM DD, YYYY`` (word-MDY, US-style) →
         ``DD/MM/YYYY``.
       * Anything unparseable → collapsed-whitespace raw.

    The normaliser is applied SYMMETRICALLY to pred and GT in
    :func:`models.pipeline_eval._nt`, so canonicalising word and
    numeric into the same representation eliminates per-format token-F1
    losses that were observed when GT used one convention and OCR
    produced another (e.g. GT ``"1 MAR 2018"`` vs pred ``"01/03/2018"``).
    """
    repaired = repair_date_ocr(value)
    m = _DATE_RE.search(repaired)
    if m is None:
        return _collapse_ws(repaired)
    raw = m.group(0).strip()
    # Numeric three-part (after repair, most dates land here).
    num = _NUM_DATE_RE.match(raw)
    if num is not None:
        d, mo, y = num.group(1), num.group(2), num.group(3)
        return f"{int(d):02d}/{int(mo):02d}/{_expand_year(y)}"
    # Word-form DMY: ``15 MAR 2018`` / ``15-MAR-2018`` / ``15/MAR/2018``.
    wdmy = _WORD_DATE_DMY_RE.match(raw)
    if wdmy is not None:
        d, mon, y = wdmy.group(1), wdmy.group(2).lower(), wdmy.group(3)
        mo = _MONTH_MAP.get(mon, mon)
        return f"{int(d):02d}/{mo}/{_expand_year(y)}"
    # Word-form MDY: ``MAR 15 2018`` / ``MAR 15, 2018`` / ``MAR-15-2018``.
    wmdy = _WORD_DATE_MDY_RE.match(raw)
    if wmdy is not None:
        mon, d, y = wmdy.group(1).lower(), wmdy.group(2), wmdy.group(3)
        mo = _MONTH_MAP.get(mon, mon)
        return f"{int(d):02d}/{mo}/{_expand_year(y)}"
    return _collapse_ws(raw)


def strip_company_registration(s: str) -> str:
    """Strip a Malaysian legal-registration suffix from a company string.

    SROIE receipts commonly OCR the company line as
    ``"GARDENIA BAKERIES KL SDN BHD 139386 X"`` while the SROIE GT is
    ``"GARDENIA BAKERIES KL SDN BHD"``.  Applied symmetrically to both
    pred and GT inside :func:`normalize_company`, the match is then
    fair — either both sides already match or both sides lose the
    suffix and align.  Up to two passes so two-segment tails
    (``SDN BHD 139386 X (M)``) are fully removed; more than two is
    unnecessary on SROIE (the longest GT drift in the 63-receipt
    test split is two suffix tokens).
    """
    for _ in range(2):
        s = _COMPANY_REG_SUFFIX.sub("", s).strip()
    return s


def normalize_company(value: str) -> str:
    r"""Collapse whitespace, strip edge + internal punctuation, repair OCR alpha.

    The SROIE GT company values are upper-case single-line strings that
    carry inconsistent trailing ``"SDN BHD."`` vs ``"SDN BHD"`` — each
    such mismatch costs a full token-F1 point.  :func:`_strip_text_punct`
    collapses ``,;:()[]."'\``` and orphan periods inside the value so
    both sides of the comparison end up with the same token set.  Case
    is preserved; token-F1 is case-insensitive already.  Digit-into-alpha
    OCR confusions are first repaired via :func:`repair_company_ocr`.

    Fix D — :func:`strip_company_registration` runs last so the
    trailing legal-registration suffix (``126926-H``, ``1248446 V``,
    ``(M)``) present on the OCR line but absent from SROIE GT does not
    cost a token-F1 point.  Applied symmetrically via :func:`_nt`
    in :mod:`models.pipeline_eval`, so a GT that *does* carry a
    suffix is stripped on both sides rather than one, keeping the
    comparison fair.
    """
    t = _strip_text_punct(repair_company_ocr(value))
    return strip_company_registration(_strip_edge_punct(_collapse_ws(t)))


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
