"""Text-prior feature extraction for the AttentionAssigner (torch-free)."""
from __future__ import annotations

import math
import re

N_TEXT_PRIORS = 6

# Local copies of the rule_based regexes so this module stays importable
# without rule_based (lightweight CI just needs the module to import).
_MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}\b")
_DATE_RE = re.compile(
    r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
    r"|\b\d{1,2}[\s/\-\.](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[\w/\-\.\s]*\d{2,4}\b",
    re.IGNORECASE,
)


def text_priors(text: str) -> list[float]:
    """Compute the 6-d text-prior feature vector for one region.

    Features:
      * ``length_log``   — log(1 + len), z-scored to ≈ [0, 1] for ≤ 400 chars.
      * ``digit_ratio``  — fraction of digits (money/date lines score high).
      * ``upper_ratio``  — fraction of uppercase letters (company header cue).
      * ``has_money``    — 1.0 if a money regex matches, else 0.0.
      * ``has_date``     — 1.0 if a date regex matches, else 0.0.
      * ``has_colon``    — 1.0 if ``:`` appears (TOTAL: / DATE: label cue).
    """
    if not text:
        return [0.0] * N_TEXT_PRIORS
    s = text.strip()
    n = len(s)
    if n == 0:
        return [0.0] * N_TEXT_PRIORS
    n_digit = sum(c.isdigit() for c in s)
    n_letter = sum(c.isalpha() for c in s)
    n_upper = sum(c.isupper() for c in s)
    return [
        math.log1p(n) / 6.0,
        n_digit / n,
        n_upper / max(n_letter, 1),
        1.0 if _MONEY_RE.search(s) else 0.0,
        1.0 if _DATE_RE.search(s) else 0.0,
        1.0 if ":" in s else 0.0,
    ]
