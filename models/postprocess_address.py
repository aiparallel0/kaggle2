"""FOCUS address normaliser — symmetric, applied to pred AND gold (Bug 18).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: closes the loss/metric gap on the address field.  Steps applied
    symmetrically to pred and gold inside
    :func:`models.normalize_bundle._normalize_address_pipeline`:
      1. Preserve line order — newlines→spaces, drop empty lines.
      2. Collapse internal whitespace runs to a single space.
      3. Strip ``, . : ;`` from non-numeric tokens (postcodes,
         lot numbers and phone tokens keep their punctuation).
      4. Casefold.
      5a. PR-ADDR-PREC-2 — leading trim: drop tax-invoice / company-
          header / OCR boilerplate tokens up to the first address
          anchor (digit-bearing token or ``no``/``lot``/``jalan``/…).
      5b. PR-ADDR-PREC — trailing trim: drop bottom-cut keywords
          (``INV``, ``CASH``, ``RECEIPT``, ``GST``, …) and 1-2-char
          alpha OCR fragments.  No-op on clean SROIE GT.
"""
from __future__ import annotations

import re

__all__ = ["normalize_address_focus"]

# Punctuation we strip from purely-alphabetic tokens.  Numeric tokens
# (``"50100"`` postcode, ``"03-1234567"`` phone, ``"12.5A"`` lot) keep
# their punctuation — it carries meaning the comparison must preserve.
_STRIP_PUNCT = ",.:;"
_DIGIT_RE = re.compile(r"\d")
_MULTI_WS_RE = re.compile(r"\s+")

# PR-ADDR-PREC — bottom-cut + company-header keywords for the trailing
# token trim.  Matches a *whole* casefolded token (after step 3 strip).
_TRAIL_DROP_TOKENS: frozenset[str] = frozenset({
    "inv", "invoice", "no", "cash", "cashier", "receipt", "tax",
    "date", "time", "doc", "bill", "roc", "tel", "telephone",
    "fax", "phone", "order", "table", "cover", "waiter", "counter",
    "credit", "note", "cashier:", "simplified",
    "bhd", "sdn", "international", "enterprise", "berhad",
    "pte", "ltd", "co", "holdings", "corp", "corporation",
    "inc", "limited", "gst", "sst", "vat",
})

# PR-ADDR-PREC-2 — leading-token strip set.  Fires from the left until
# the first :data:`_LEAD_ANCHOR_TOKENS` member or digit-bearing token
# (postcode / lot / house no.).  When the entire token stream contains
# no anchor at all the leading trim is a no-op so existing tests like
# ``"abc def" -> "abc def"`` keep their pre-PR shape.
_LEAD_DROP_TOKENS: frozenset[str] = frozenset({
    "tax", "invoice", "simplified", "receipt",
    "sdn", "bhd", "berhad", "pte", "ltd", "co", "co.",
    "holdings", "corp", "corporation", "inc", "inc.", "limited",
    "international", "enterprise", "enterprises", "trading", "marketing",
    "welcome", "to", "thank", "you",
    "gst", "sst", "vat", "roc",
    "and", "the", "&",
})

# Address-anchor keywords — a leading token in this set halts the
# trim.  Malaysian / Singaporean address vocabulary plus generic
# building-name nouns; digit-bearing tokens are anchors via _DIGIT_RE.
_LEAD_ANCHOR_TOKENS: frozenset[str] = frozenset({
    "no", "lot", "block", "unit", "level", "floor", "ground",
    "lower", "upper", "lg", "lg-",
    "jalan", "lorong", "persiaran", "taman", "kawasan", "kampung",
    "batu", "jln", "kg", "tmn", "blk",
    "plaza", "mall", "tower", "complex", "centre", "center",
    "building", "bangunan",
})

_TAIL_FRAG_MAX_LEN = 2

# PR-ADDR-DEDUPE — span lengths to scan for consecutive repeats.  Run
# 20260430T125211Z surfaced the assigner double-attention failure mode
# where a token-window appears twice in a row in the predicted address
# ("bandar bukit raja 41050 bandar bukit raja 41050 klang"), the n=91
# wrong_span class for pipeline address.  Symmetric (applied to both
# pred and GT) so a clean GT remains a fixed point.  Spans tried in
# decreasing order so the longest repeat wins on overlap.
_DEDUPE_SPANS: tuple[int, ...] = (5, 4, 3, 2)


def _strip_token_punct(token: str) -> str:
    """Strip ``,.:;`` from a token unless it carries any digit."""
    if _DIGIT_RE.search(token):
        return token
    return token.translate(str.maketrans("", "", _STRIP_PUNCT))


def _is_lead_anchor(token: str) -> bool:
    """True iff ``token`` is digit-bearing or an address-anchor word."""
    if not token:
        return False
    if _DIGIT_RE.search(token):
        return True
    bare = token.rstrip(",.:;'")
    return bare in _LEAD_ANCHOR_TOKENS


def _trim_leading_junk(tokens: list[str]) -> list[str]:
    """Drop leading company-header tokens up to the first address anchor.

    Walks left-to-right looking for the first :func:`_is_lead_anchor`
    token; when such a token exists at position ``k > 0`` and every
    preceding token is either in :data:`_LEAD_DROP_TOKENS` or a purely-
    alphabetic OCR fragment, the head is excised.  When no anchor is
    found, the input is returned unchanged so existing tests like
    ``"abc def" -> "abc def"`` keep their pre-PR shape.
    """
    if not tokens:
        return tokens
    anchor = next(
        (k for k, t in enumerate(tokens) if _is_lead_anchor(t)), None,
    )
    if anchor is None or anchor == 0:
        return list(tokens)
    for tok in tokens[:anchor]:
        if not tok or tok in _LEAD_DROP_TOKENS or tok.isalpha():
            continue
        return list(tokens)
    return list(tokens[anchor:])


def _trim_trailing_junk(tokens: list[str]) -> list[str]:
    """Drop trailing bottom-cut keywords and 1-2-char alpha fragments.

    Walks right-to-left, removing any token in :data:`_TRAIL_DROP_TOKENS`
    or any 1-2-char purely-alphabetic OCR fragment.  Halts at the first
    digit-bearing token (postcode / lot / lot-number — the canonical
    *end* of a Malaysian address) or non-droppable alpha token.
    """
    out = list(tokens)
    while out:
        tail = out[-1]
        if not tail:
            out.pop()
            continue
        if _DIGIT_RE.search(tail):
            break
        if tail in _TRAIL_DROP_TOKENS:
            out.pop()
            continue
        if len(tail) <= _TAIL_FRAG_MAX_LEN and tail.isalpha():
            out.pop()
            continue
        break
    return out


def _dedupe_consecutive_runs(tokens: list[str]) -> list[str]:
    """Drop consecutive repeated 2-to-5-token spans.

    Scans for ``[A B C][A B C]`` and reduces it to ``[A B C]``.  Spans
    tried longest-first so an overlapping pair like
    ``[a b c d][a b c d]`` is not split into two 2-token dedupes.
    Idempotent: a clean address has no repeats and is returned unchanged.
    """
    out = list(tokens)
    for span in _DEDUPE_SPANS:
        i = 0
        while i + 2 * span <= len(out):
            if out[i:i + span] == out[i + span:i + 2 * span]:
                del out[i + span:i + 2 * span]
            else:
                i += 1
    return out


def normalize_address_focus(value: str) -> str:
    """Symmetric address normaliser — line order preserved, casefold output.

    No-op on the empty string.  Output is whitespace-collapsed,
    case-folded, with comma/period/colon/semicolon stripped from non-
    numeric tokens, leading boilerplate (PR-ADDR-PREC-2) and trailing
    bottom-cut / 1-2-char OCR fragments (PR-ADDR-PREC) trimmed.
    """
    if not value:
        return ""
    lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
    joined = " ".join(lines) if lines else value
    collapsed = _MULTI_WS_RE.sub(" ", joined).strip()
    tokens = [_strip_token_punct(t) for t in collapsed.split(" ")]
    tokens = [t for t in tokens if t]
    folded = [t.casefold() for t in tokens]
    # Step 5a (PR-ADDR-PREC-2): leading company-header / tax-invoice trim.
    # Step 5b (PR-ADDR-PREC):   trailing bottom-cut + 1-2-char OCR trim.
    # Step 6  (PR-ADDR-DEDUPE): collapse consecutive repeated token spans.
    trimmed = _trim_trailing_junk(_trim_leading_junk(folded))
    return " ".join(_dedupe_consecutive_runs(trimmed))
