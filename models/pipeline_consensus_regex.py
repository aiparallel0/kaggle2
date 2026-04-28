"""PR-A / T-D2 — regex half of ``pipeline_consensus`` as its own module.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: namespace home for the regex constants used by the consensus
    refiner (``_TOTAL_*_RE``, ``_ADDR_*_RE``, etc.).  These already
    live in :mod:`models.rule_regex` and (locally) in
    :mod:`models.pipeline_consensus`; this module re-exports them at
    a single import site so callers can ``from
    models.pipeline_consensus_regex import _TOTAL_STRONG, _ADDR_ANCHOR``.
"""
from __future__ import annotations

import re

from models.pipeline_consensus import _CURRENCY_CUE_RE, _CURRENCY_PREFIX_RE
from models.rule_regex import (
    _ADDR_ANCHOR,
    _ADDR_CONTINUATION,
    _ADDR_EXCLUDE,
    _ADDR_LEADING_JUNK_RE,
    _ADDR_TERMINATOR,
    _COMPANY_TOKEN,
    _DATE_RE,
    _HEADER_JUNK,
    _MONEY_RE,
    _POSTCODE_RE,
    _TOTAL_NEGATIVE,
    _TOTAL_STRONG,
    _TOTAL_WEAK,
    repair_money_ocr,
)

# PR-C / S2 — Bahasa-aware total / subtotal / distractor patterns
# co-located with the other regex constants so the consensus refiner
# can pick up the priors_v3 keyword set without re-importing from
# attention_priors.
_TOTAL_KW = re.compile(
    r"\b(grand\s*total|jumlah\s*besar|total)\b", re.IGNORECASE,
)
_SUBTOTAL_KW = re.compile(r"\bsub[\- ]?total\b", re.IGNORECASE)
_DISTRACTOR_KW = re.compile(
    r"\b(cash|change|tax|due|rounding|balance|gst|svc|tip|"
    r"discount|tunai|baki)\b",
    re.IGNORECASE,
)

__all__ = [
    "_ADDR_ANCHOR",
    "_ADDR_CONTINUATION",
    "_ADDR_EXCLUDE",
    "_ADDR_LEADING_JUNK_RE",
    "_ADDR_TERMINATOR",
    "_COMPANY_TOKEN",
    "_CURRENCY_CUE_RE",
    "_CURRENCY_PREFIX_RE",
    "_DATE_RE",
    "_DISTRACTOR_KW",
    "_HEADER_JUNK",
    "_MONEY_RE",
    "_POSTCODE_RE",
    "_SUBTOTAL_KW",
    "_TOTAL_KW",
    "_TOTAL_NEGATIVE",
    "_TOTAL_STRONG",
    "_TOTAL_WEAK",
    "repair_money_ocr",
]
