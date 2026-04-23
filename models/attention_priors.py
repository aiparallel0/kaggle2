"""Handcrafted 6/9/14-d text-prior features for the AttentionAssigner.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: computes prior vectors that augment TrOCR features in the
    AttentionAssigner's region encoder.  Three versions ship:
      v1 (6-d): length_log, digit_ratio, upper_ratio,
                has_money, has_date, has_colon
      v2 (9-d): v1 + has_total_keyword, is_last_money_line, y_norm
      v3 (14-d): v2 + is_subtotal, is_cash, is_change, is_tax,
                is_rounding  — distractor-aware bits that bypass the
                mean-pool bottleneck for the hardest confusers
                (strategy E of the assigner plan).
    No torch dependency.
"""
from __future__ import annotations

import math
import re

N_TEXT_PRIORS = 6
N_TEXT_PRIORS_V2 = 9
N_TEXT_PRIORS_V3 = 14

# Local copies of the rule_based regexes so this module stays importable
# without rule_based (lightweight CI just needs the module to import).
_MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}\b")
_DATE_RE = re.compile(
    r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
    r"|\b\d{1,2}[\s/\-\.](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[\w/\-\.\s]*\d{2,4}\b",
    re.IGNORECASE,
)
_TOTAL_KW = re.compile(r"\b(total|amount|grand|due|payable)\b", re.IGNORECASE)
_SUBTOTAL_KW = re.compile(r"\bsub[\s\-]?total\b|\bsubtotal\b", re.IGNORECASE)

# Distractor keywords for the v3 priors (strategy E).  Kept narrow so a
# single bit is an unambiguous "this line is NOT the grand-total".
_CASH_KW = re.compile(r"\b(cash(?:\s+tendered)?|tendered|kembalian)\b", re.IGNORECASE)
_CHANGE_KW = re.compile(r"\bchange\b", re.IGNORECASE)
_TAX_KW = re.compile(r"\b(tax|gst|sst|vat)\b", re.IGNORECASE)
_ROUNDING_KW = re.compile(r"\bround(?:ing|ed)?\b", re.IGNORECASE)
# Currency prefix on the same line as a money value — a strong GRAND-TOTAL
# cue (mirrors ``_CURRENCY_CUE_RE`` in pipeline_consensus).
_CURRENCY_CUE = re.compile(r"\b(?:RM|MYR|\$)\b", re.IGNORECASE)


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


def text_priors_v2(text: str, y_norm: float, is_last_money: bool) -> list[float]:
    """9-d priors: base 6-d + has_total_keyword, is_last_money_line, y_norm."""
    base = text_priors(text)
    s = text.strip()
    has_total = (
        1.0 if _TOTAL_KW.search(s) and not _SUBTOTAL_KW.search(s) else 0.0
    )
    return base + [has_total, 1.0 if is_last_money else 0.0, float(y_norm)]


def text_priors_v3(
    text: str, y_norm: float, is_last_money: bool,
) -> list[float]:
    """14-d priors: base 9-d + 5 distractor-aware bits (strategy E).

    The extra bits (``is_subtotal``, ``is_cash``, ``is_change``,
    ``is_tax``, ``is_rounding``) bypass the TrOCR mean-pool bottleneck
    that erases the SUBTOTAL-vs-TOTAL sub-word signal.  Feeding them
    through ``prior_proj`` instead of the text encoder gives the
    assigner direct, per-region access to the canonical confusers from
    the live miss table.  The ``_CURRENCY_CUE`` bit is intentionally
    *not* added because the same signal is already captured by
    ``_score_money`` and including it here would double-count.
    """
    base = text_priors_v2(text, y_norm, is_last_money)
    s = text.strip()
    return base + [
        1.0 if _SUBTOTAL_KW.search(s) else 0.0,
        1.0 if _CASH_KW.search(s) else 0.0,
        1.0 if _CHANGE_KW.search(s) else 0.0,
        1.0 if _TAX_KW.search(s) else 0.0,
        1.0 if _ROUNDING_KW.search(s) else 0.0,
    ]


__all__ = [
    "N_TEXT_PRIORS",
    "N_TEXT_PRIORS_V2",
    "N_TEXT_PRIORS_V3",
    "text_priors",
    "text_priors_v2",
    "text_priors_v3",
    "_MONEY_RE",
    "_DATE_RE",
    "_CURRENCY_CUE",
]
