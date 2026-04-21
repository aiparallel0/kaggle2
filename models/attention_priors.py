"""Handcrafted 6-d text-prior features for the AttentionAssigner.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: computes the 6-d prior vector (length_log, digit_ratio, upper_ratio,
    has_money, has_date, has_colon) that augments TrOCR features in the
    AttentionAssigner's region encoder.  No torch dependency.
"""
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
    """6-d text-prior features: length, digit/upper ratio, money/date/colon."""
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
