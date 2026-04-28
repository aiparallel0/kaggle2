"""Handcrafted 6/9/14/20-d text-prior features for the AttentionAssigner.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: computes prior vectors that augment TrOCR features in the
    AttentionAssigner's region encoder.  Four versions ship:
      v1 (6-d): length_log, digit_ratio, upper_ratio,
                has_money, has_date, has_colon
      v2 (9-d): v1 + has_total_keyword, is_last_money_line, y_norm
      v3 (14-d): v2 + is_subtotal, is_cash, is_change, is_tax,
                is_rounding  — distractor-aware bits that bypass the
                mean-pool bottleneck for the hardest confusers
                (strategy E of the assigner plan).
      v4 (20-d): v3 + is_subtotal_kw, is_tax_kw, is_company_boilerplate,
                line_y_normalised, money_value_normalised,
                arithmetic_witness_self  — FOCUS-T/C structural priors
                (paper §III-D rewrite, FOCUS framework).
    No torch dependency.
"""
from __future__ import annotations

import math
import re

N_TEXT_PRIORS = 6
N_TEXT_PRIORS_V2 = 9
N_TEXT_PRIORS_V3 = 14
# FOCUS-T/C — priors_v4 appends 6 dims to v3 (FOCUS framework, paper §III-D
# rewrite).  The first two duplicate v3 bits (``is_subtotal_kw`` /
# ``is_tax_kw``) but are kept as explicit named columns so the FOCUS-T head's
# ``arithmetic_witness_self`` provenance is auditable from the prior tensor
# alone — the spec lists all six dims even though two are redundant with v3.
#   [14] is_subtotal_kw                 (== v3[9],  reserved-for-clarity)
#   [15] is_tax_kw                      (== v3[12], reserved-for-clarity)
#   [16] is_company_boilerplate         (regex: SDN BHD / BERHAD / PTE LTD / INC)
#   [17] line_y_normalised              (y_top / image_height; mirrors v3[8])
#   [18] money_value_normalised         (money / max_money_on_receipt; 0 if none)
#   [19] arithmetic_witness_self        (1 iff line_money == sub+tax pair, ε=2¢)
N_TEXT_PRIORS_V4 = 20
V4_IS_SUBTOTAL_KW_IDX = 14
V4_IS_TAX_KW_IDX = 15
V4_IS_COMPANY_BOILERPLATE_IDX = 16
V4_Y_NORM_IDX = 17
V4_MONEY_NORM_IDX = 18
V4_WITNESS_IDX = 19

# Local copies of the rule_based regexes so this module stays importable
# without rule_based (lightweight CI just needs the module to import).
_MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}\b")
_DATE_RE = re.compile(
    r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
    r"|\b\d{1,2}[\s/\-\.](?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
    r"[\w/\-\.\s]*\d{2,4}\b",
    re.IGNORECASE,
)
_TOTAL_KW = re.compile(
    r"\b(grand\s*total|total|jumlah|amount(?:\s+due)?|nett?|due|payable)\b",
    re.IGNORECASE,
)
# Fix 2 — the SUBTOTAL bit is widened to ``service`` (which is an
# itemised line-item, never the grand total) but intentionally NOT to
# ``tender`` / ``change`` / ``rounding``: the v3 prior vector already
# has dedicated :data:`_CASH_KW` / :data:`_CHANGE_KW` / :data:`_ROUNDING_KW`
# bits for those, and re-adding them here would double-count a region's
# distractor evidence inside the assigner's prior projection.
_SUBTOTAL_KW = re.compile(
    r"\bsub[\s\-]?total\b|\bsubtotal\b|\bservice\s+charge\b|\bservice\s+tax\b",
    re.IGNORECASE,
)

# Distractor keywords for the v3 priors (strategy E).  Kept narrow so a
# single bit is an unambiguous "this line is NOT the grand-total".
_CASH_KW = re.compile(r"\b(cash(?:\s+tendered)?|tendered|kembalian)\b", re.IGNORECASE)
_CHANGE_KW = re.compile(r"\bchange\b", re.IGNORECASE)
_TAX_KW = re.compile(r"\b(tax|gst|sst|vat)\b", re.IGNORECASE)
_ROUNDING_KW = re.compile(r"\bround(?:ing|ed)?\b", re.IGNORECASE)
# FOCUS-C — Malaysian/Singaporean/Indian/US legal-entity suffixes. These
# regexes match a "boilerplate" company-suffix line so the FOCUS-C head can
# down-weight ``"GROCER MART SDN BHD"``-style suffix lines and select the
# trade-name line above instead.
_COMPANY_BOILERPLATE_KW = re.compile(
    r"\b(?:SDN[\s\-]?BHD|BERHAD|PTE[\s\-]?LTD|PTY[\s\-]?LTD|"
    r"LLC|LLP|INC\.?|CORP\.?|LIMITED|LTD\.?|GMBH|BHD)\b",
    re.IGNORECASE,
)
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


def _parse_money(text: str) -> float | None:
    """Return the *first* parseable ``\\d+,\\d{3}\\.\\d{2}`` money value on
    ``text`` (or any plain ``\\d+\\.\\d{2}``), as a float in receipt currency.
    Returns ``None`` when no money token is present.  Used by the FOCUS-T
    arithmetic-witness builder; kept private to this module.
    """
    m = _MONEY_RE.search(text)
    if m is None:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def arithmetic_witnesses_v4(
    texts: list[str], eps: float = 0.02,
) -> list[float]:
    """O(N²) per-receipt witness column for priors_v4 (FOCUS-T).

    For each line ``i`` carrying a money value ``mi`` and *no* SUBTOTAL/TAX
    keyword (i.e. a candidate GRAND-TOTAL line), set ``witness[i] = 1.0``
    iff there exists some ``j`` carrying a SUBTOTAL keyword + value ``mj``
    AND some ``k`` carrying a TAX keyword + value ``mk`` such that
    ``|mi - (mj + mk)| <= eps``.  Lines without money, or that themselves
    look like SUBTOTAL/TAX, get ``0.0``.  The receipt-level computation is
    done once and the column is broadcast to all regions.
    """
    n = len(texts)
    monies: list[float | None] = [_parse_money(t) for t in texts]
    is_sub = [bool(_SUBTOTAL_KW.search(t)) for t in texts]
    is_tax = [bool(_TAX_KW.search(t)) for t in texts]
    out = [0.0] * n
    for i in range(n):
        mi = monies[i]
        if mi is None or is_sub[i] or is_tax[i]:
            continue
        for j in range(n):
            mj = monies[j]
            if not is_sub[j] or mj is None:
                continue
            for k in range(n):
                mk = monies[k]
                if k == j or not is_tax[k] or mk is None:
                    continue
                if abs(mi - (mj + mk)) <= eps:
                    out[i] = 1.0
                    break
            if out[i] == 1.0:
                break
    return out


def text_priors_v4(
    text: str, y_norm: float, is_last_money: bool,
    money_value_norm: float, witness_self: float,
) -> list[float]:
    """20-d FOCUS framework priors: v3 (14-d) + 6 FOCUS-T/C dims (indices
    :data:`V4_IS_SUBTOTAL_KW_IDX` .. :data:`V4_WITNESS_IDX`).

    The first two appended dims duplicate v3 bits but are kept named so the
    FOCUS-T arithmetic-witness derivation is self-contained inside the
    priors tensor (i.e. the FOCUS heads never have to re-index across v3 /
    v4 to find a column).  ``money_value_norm`` and ``witness_self`` are
    receipt-level quantities the caller computes once and broadcasts.
    """
    base = text_priors_v3(text, y_norm, is_last_money)
    s = text.strip()
    return base + [
        1.0 if _SUBTOTAL_KW.search(s) else 0.0,
        1.0 if _TAX_KW.search(s) else 0.0,
        1.0 if _COMPANY_BOILERPLATE_KW.search(s) else 0.0,
        float(y_norm),
        float(money_value_norm),
        float(witness_self),
    ]


__all__ = [
    "N_TEXT_PRIORS",
    "N_TEXT_PRIORS_V2",
    "N_TEXT_PRIORS_V3",
    "N_TEXT_PRIORS_V4",
    "V4_IS_COMPANY_BOILERPLATE_IDX",
    "V4_IS_SUBTOTAL_KW_IDX",
    "V4_IS_TAX_KW_IDX",
    "V4_MONEY_NORM_IDX",
    "V4_WITNESS_IDX",
    "V4_Y_NORM_IDX",
    "arithmetic_witnesses_v4",
    "text_priors",
    "text_priors_v2",
    "text_priors_v3",
    "text_priors_v4",
    "_MONEY_RE",
    "_DATE_RE",
    "_CURRENCY_CUE",
    "_COMPANY_BOILERPLATE_KW",
]
