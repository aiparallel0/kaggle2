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


def _strip_token_punct(token: str) -> str:
    """Strip ``,.:;`` from a token unless it carries any digit."""
    if _DIGIT_RE.search(token):
        return token
    return token.translate(str.maketrans("", "", _STRIP_PUNCT))


def normalize_address_focus(value: str) -> str:
    """Symmetric address normaliser — line order preserved, casefold output.

    A no-op on the empty string.  The output is whitespace-collapsed,
    case-folded, and has comma/period/colon/semicolon stripped from
    non-numeric tokens — the token punctuation that the SROIE GT
    convention is inconsistent about, but which token-F1 treats as a
    full token loss when one side carries it and the other does not.
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
    # Step 4 — casefold for case-insensitive comparison parity.  The
    # legacy ``normalize_address`` returned mixed case; downstream
    # ``compute_field_f1`` already lower-cases for matching, so this
    # change is symmetric on both sides of the comparison.
    return " ".join(tokens).casefold()
