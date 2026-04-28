"""FOCUS address normaliser — symmetric, applied to pred AND gold (Bug 18).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: closes the loss/metric gap on the address field.  The deterministic
    numeric normaliser was previously applied only to ``total``; address
    matched verbatim.  Comma/period/casing drift on lines whose line set
    was already correct therefore destroyed token-F1 and (especially) EM.

    The function applied symmetrically to BOTH pred and gold inside
    :func:`models.eval_pipeline._nt`, mirroring the existing total
    normaliser's wiring.  Steps (in order):

      1. Preserve line order — split on newlines, drop empty lines.
         The assembly layer (``focus_pipeline``) emits address lines in
         y_min-ascending order from the predicted span; we keep that
         order rather than re-sorting (string-only normaliser, no bbox
         info available at this layer).  GT is single-line on SROIE,
         so the multi-line case is rare and harmless.
      2. Collapse whitespace runs to a single space.
      3. Strip ``, . : ;`` from non-numeric tokens.  Tokens matching
         ``\\d`` (postcodes ``50100``, phone fragments, lot numbers
         ``Lot 12.5A``) keep their punctuation so the postcode and
         the unit-number convention are preserved.
      4. Casefold.
      5. PR-ADDR-PREC — token-level *trailing* trim: drop trailing
         tokens that match the bottom-cut keyword set (``INV``, ``NO``,
         ``CASH``, ``RECEIPT``, ``TAX``, ``INVOICE``, ``DATE``,
         ``TIME``, ``DOC``, ``BILL``, ``ROC``, ``TEL``, ``FAX``,
         ``BHD``, ``SDN``, ``INTERNATIONAL``, ``ENTERPRISE``, …) or
         are 1-2-character OCR fragments (``JO``, ``T``, ``#``).  Run
         symmetrically on pred and GT — the SROIE GT is clean of
         these tokens so the trim is a no-op for gold but excises the
         tail-bleed seen on 331/347 mismatched receipts in
         ``address_mismatches.json``.
"""
from __future__ import annotations

import re

__all__ = ["normalize_address_focus"]

# Punctuation we strip from purely-alphabetic tokens.  Numeric tokens
# (``"50100"`` postcode, ``"03-1234567"`` phone, ``"12.5A"`` lot) are
# left intact — their punctuation carries meaning the comparison must
# preserve.
_STRIP_PUNCT = ",.:;"
_DIGIT_RE = re.compile(r"\d")
_MULTI_WS_RE = re.compile(r"\s+")

# PR-ADDR-PREC — bottom-cut + company-header keywords for the trailing
# token trim.  Matches a *whole* casefolded token (after step 3 strip).
# Kept as a Python set (frozenset for immutability) so the trim is a
# membership test rather than yet another regex compile.
_TRAIL_DROP_TOKENS: frozenset[str] = frozenset({
    # bottom-cut transaction boundary tokens
    "inv", "invoice", "no", "cash", "cashier", "receipt", "tax",
    "date", "time", "doc", "bill", "roc", "tel", "telephone",
    "fax", "phone", "order", "table", "cover", "waiter", "counter",
    "credit", "note", "cashier:", "simplified",
    # top-of-receipt company / tax-ID stripping tokens
    "bhd", "sdn", "international", "enterprise", "berhad",
    "pte", "ltd", "co", "holdings", "corp", "corporation",
    "inc", "limited", "gst", "sst", "vat",
})

# 1-2-char OCR fragments at the tail are dropped unless they are
# digit-bearing (so postcodes, lot numbers, and ``#3`` survive — the
# trim only fires on alpha fragments / lone punctuation).
_TAIL_FRAG_MAX_LEN = 2


def _strip_token_punct(token: str) -> str:
    """Strip ``,.:;`` from a token unless it carries any digit."""
    if _DIGIT_RE.search(token):
        return token
    return token.translate(str.maketrans("", "", _STRIP_PUNCT))


def _trim_trailing_junk(tokens: list[str]) -> list[str]:
    """Drop trailing bottom-cut keywords and 1-2-char alpha fragments.

    Walks the token list right-to-left and removes any token that is
    either a member of :data:`_TRAIL_DROP_TOKENS` (case-folded) OR is
    a short (≤2-char) purely-alphabetic OCR fragment.  Stops at the
    first token that does NOT meet either condition so the head of
    the address is never touched.  Numeric / digit-bearing tokens
    (``50100``, ``12.5a``) always halt the trim — postcodes are the
    canonical *end* of a Malaysian address.
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


def normalize_address_focus(value: str) -> str:
    """Symmetric address normaliser — line order preserved, casefold output.

    A no-op on the empty string.  The output is whitespace-collapsed,
    case-folded, has comma/period/colon/semicolon stripped from
    non-numeric tokens, and has trailing bottom-cut / 1-2-char OCR
    fragments dropped (PR-ADDR-PREC).  Applied symmetrically to pred
    and GT inside :func:`models.normalize_bundle._normalize_address_pipeline`.
    """
    if not value:
        return ""
    # Step 1 — preserve line order while joining; SROIE GT is single
    # line, so this collapses to a no-op on most receipts.
    lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
    joined = " ".join(lines) if lines else value
    # Step 2 — collapse internal whitespace runs.
    collapsed = _MULTI_WS_RE.sub(" ", joined).strip()
    # Step 3 — per-token punctuation strip (numeric tokens preserved).
    tokens = [_strip_token_punct(t) for t in collapsed.split(" ")]
    # Drop tokens that became empty after punctuation strip (a lone
    # ``","`` or ``"."`` between alpha tokens).
    tokens = [t for t in tokens if t]
    # Step 4 — casefold for case-insensitive comparison parity.
    folded = [t.casefold() for t in tokens]
    # Step 5 — PR-ADDR-PREC — drop trailing bottom-cut keywords and
    # short alpha OCR fragments so company headers / inv-no / cash-
    # receipt / doc-no tails don't leak into the predicted span.
    trimmed = _trim_trailing_junk(folded)
    return " ".join(trimmed)
