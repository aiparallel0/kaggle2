"""Regex patterns for the rule-based KIE baseline (no ML dependencies).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: DATE_RE and MONEY_RE are the primary field detectors; _TOTAL_NEGATIVE,
    _TOTAL_STRONG, _TOTAL_WEAK drive the keyword ranking in extract_total.
"""
from __future__ import annotations

import re

# DATE_RE accepts:
#   2024/08/01, 01-08-2024, 01.08.2019, 1-8-24   (numeric, 3-part)
#   01-AUG-2019, 1 AUG 2019, 1/AUG/2019          (word month between digits)
#   August 1, 2019, AUG 01 2019                  (word month first)
_NUM_DATE = r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_WORD_DATE = (
    rf"\b\d{{1,2}}[\s/\-\.]{_MONTHS}[\s/\-\.]\d{{2,4}}\b"
    rf"|\b{_MONTHS}[\s/\-\.]\d{{1,2}}[,\s/\-\.]+\d{{2,4}}\b"
)
DATE_RE = re.compile(f"(?:{_NUM_DATE})|(?:{_WORD_DATE})", re.IGNORECASE)

# Money — matches ``12.30``, ``$12.30``, ``RM12.30``, ``1,234.56``.
MONEY_RE = re.compile(r"(?:RM|USD|SGD|MYR|\$)?\s*\d{1,3}(?:,\d{3})*\.\d{2}\b", re.IGNORECASE)

# OCR-confused digit spans: a numeric-looking run where TrOCR may have
# substituted ``O``/``o`` for ``0``, ``l``/``I`` for ``1``, or the European
# ``,`` for the decimal ``.``.  The span must start and end with a digit or
# an OCR-confused digit so we don't pick up pure-letter tokens like ``IO``.
_MONEY_OCR_SPAN = re.compile(r"[\dOolI][\dOolI.,]*[\dOolI]")


def repair_money_ocr(s: str) -> str:
    """Fix common TrOCR money OCR errors inside digit-only spans.

    Substitutes ``O``/``o``→``0`` and ``l``/``I``→``1`` inside spans that
    already look numeric, and converts a lone ``,`` decimal separator to
    ``.`` (European → US format) so ``MONEY_RE`` can match ``43,50`` or
    ``43.5O`` the same way it matches ``43.50``.  Non-numeric tokens are
    left untouched because the span anchors require a digit-like boundary.
    """
    def _fix(m: re.Match[str]) -> str:
        t = m.group(0).translate(str.maketrans("Ool", "001")).replace("I", "1")
        # The ``,``→``.`` swap handles the European-format case
        # (single comma, no existing dot) so ``43,50`` parses as ``43.50``.
        if t.count(",") == 1 and "." not in t:
            t = t.replace(",", ".")
        return t
    return _MONEY_OCR_SPAN.sub(_fix, s)


# Backwards-compatible aliases.
_DATE_RE = DATE_RE
_MONEY_RE = MONEY_RE

# Words that disqualify a money region from being TOTAL.
_TOTAL_NEGATIVE = re.compile(
    r"\b(sub\s*-?\s*total|subtotal|sub|round(?:ing|ed)?|"
    r"change|cash\s+tendered|tendered|balance|credit|debit|"
    r"card|visa|master(?:card)?|paid|payment|kembalian|"
    r"discount|service|charge|tax\s+(?:only|\d)|gst\s+\d|sst\s+\d|"
    r"qty|item|no\.)\b",
    re.IGNORECASE,
)

# Positive TOTAL cues — strongest signal is "GRAND TOTAL" > "TOTAL" > "AMOUNT".
_TOTAL_STRONG = re.compile(
    r"\b(grand\s*total|amount\s*(?:due|payable)|nett?\s*total|total\s*(?:due|amt|amount))\b",
    re.IGNORECASE,
)
_TOTAL_WEAK = re.compile(r"\btotal\b|\bamount\b", re.IGNORECASE)

# Header junk the top-line heuristic should skip when picking a company.
_HEADER_JUNK = re.compile(
    r"^\s*(tax\s*invoice|invoice|receipt|cash\s*(?:sale|bill)|"
    r"bill|original(?:\s*copy)?|copy|reprint|duplicate|"
    r"welcome|thank\s*you|customer\s*copy|merchant\s*copy)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Phone / tax ID / reg-no patterns that often sneak into the address block.
_ADDR_EXCLUDE = re.compile(
    r"\b(tel(?:ephone|\.?)?|phone|fax|h\/?p|hp|mobile|email|e-mail|"
    r"gst|sst|reg(?:istration)?(?:\s*no\.?)?|co(?:mpany)?\s*no\.?|"
    r"kad|vat|tin|www\.|http|\.com|\.my)\b",
    re.IGNORECASE,
)
