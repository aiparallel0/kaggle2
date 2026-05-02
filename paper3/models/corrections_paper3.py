"""Character-level OCR corrections for the assignment stage.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: recovers from common TrOCR character confusions inside field-type-
    aware sub-strings so the assignment stage can *lift* pipeline F1
    above the naive product of per-component F1s (0.97·0.97·0.97 = 0.91)
    by doing work that YOLO/TrOCR cannot.

    Confusions this module repairs (kept conservative so alphabetic
    tokens are not mangled):
      * ``O`` / ``o``  → ``0``  in digit-only spans
      * ``l`` / ``I``  → ``1``  in digit-only spans
      * ``S`` / ``s``  → ``5``  in digit-only spans
      * ``B``         → ``8``  in digit-only spans
      * ``Z``         → ``2``  in digit-only spans
      * ``,``         → ``.``  as European decimal separator
      * ``8``         → ``B``  when inside a pure-alpha word (company)

    Per-field entry points:
      * :func:`repair_date_ocr`     — splits 8-digit runs into DD/MM/YYYY,
                                      canonicalises separators, repairs
                                      digit confusions in date spans.
      * :func:`repair_postcode_ocr` — 5-digit postcode span repair inside
                                      free-form address strings.
      * :func:`repair_company_ocr` — letter-only word repair (``SDN 8HD``
                                      → ``SDN BHD``).
"""
from __future__ import annotations

import re

__all__ = [
    "repair_company_ocr",
    "repair_date_ocr",
    "repair_postcode_ocr",
]

# Digit-run substitution table used inside all three repairers.
_DIGIT_SUBS = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1",
                             "S": "5", "s": "5", "B": "8", "Z": "2"})

# A run that *could* be digits: digits or their common confusables.
_DIGITISH = re.compile(r"[0-9OolISsBZ][0-9OolISsBZ.,/\-]*[0-9OolISsBZ]")


def _repair_digit_run(m: re.Match[str]) -> str:
    """Apply digit-only subs inside a single numeric run; never over-convert."""
    return m.group(0).translate(_DIGIT_SUBS)


# ---------------------------------------------------------------- DATE ----

_DATE_SEPS_RE = re.compile(r"[\-\.]")
_DATE_FULL_RE = re.compile(
    r"\b(\d{1,2})[\-/\.](\d{1,2})[\-/\.](\d{2,4})\b",
)
_DATE_COMPACT_RE = re.compile(r"\b(\d{2})(\d{2})(\d{2,4})\b")


def repair_date_ocr(value: str) -> str:
    """Reconstruct a canonical ``DD/MM/YYYY`` from noisy OCR date text.

    Handles compact runs ``12032026`` → ``12/03/2026``, any separator
    variant (``.``, ``-``, ``/``) → ``/``, and digit confusions (``Ol``
    → ``01``, ``l2`` → ``12``) when the run is digit-shaped.  If nothing
    looks like a date we return the input unchanged so the downstream
    regex fallback still has a chance.
    """
    # First, repair digit-confusions inside digit-shaped runs.
    repaired = _DIGITISH.sub(_repair_digit_run, value)
    # Try dotted/dashed date → slashed.
    m = _DATE_FULL_RE.search(repaired)
    if m is not None:
        d, mo, y = m.groups()
        return f"{int(d):02d}/{int(mo):02d}/{y}"
    # Try compact 6/8-digit run (DDMMYY / DDMMYYYY).
    c = _DATE_COMPACT_RE.search(repaired)
    if c is not None:
        d, mo, y = c.groups()
        # Bounds check so we don't mutilate phone numbers that happen to
        # contain 8 consecutive digits.
        if 1 <= int(d) <= 31 and 1 <= int(mo) <= 12:
            return f"{int(d):02d}/{int(mo):02d}/{y}"
    return repaired


# ------------------------------------------------------------ POSTCODE ----

_POSTCODE_RUN_RE = re.compile(r"\b[0-9OolISsBZ]{5}\b")


def repair_postcode_ocr(value: str) -> str:
    """Repair digit confusions inside 5-digit postcode-shaped runs only.

    SROIE is Malaysia-majority and uses 5-digit postcodes (e.g.
    ``50100 KUALA LUMPUR``); TrOCR commonly drops ``0``/``O``/``l``/``I``.
    Matching is word-boundary-anchored so street numbers and house
    numbers (``NO. 12``) are not touched.
    """
    def _fix(m: re.Match[str]) -> str:
        run = m.group(0)
        fixed = run.translate(_DIGIT_SUBS)
        # Only rewrite when the repaired run is actually digits now.
        return fixed if fixed.isdigit() else run
    return _POSTCODE_RUN_RE.sub(_fix, value)


# -------------------------------------------------------------- COMPANY ----

# Alpha-looking token that contains at least one digit-confusable.
# Only convert ``8→B``/``0→O`` in tokens that are otherwise alphabetic
# so we never mangle ``BLOCK 3`` or ``LOT 8``.
_ALPHA_TOKEN_RE = re.compile(r"\b[A-Z0-9]*[A-Z][A-Z0-9]*\b")
_ALPHA_SUBS = str.maketrans({"8": "B", "0": "O", "1": "I", "5": "S"})


def repair_company_ocr(value: str) -> str:
    """Repair digit-into-alpha confusions inside pure-alpha tokens only.

    Converts e.g. ``SDN 8HD`` → ``SDN BHD``, ``5DN 8HD`` → ``SDN BHD``,
    but leaves ``BLOCK 3`` / ``LOT 8A`` / ``NO. 1`` untouched because
    they are not mostly-alpha tokens.  Decision rule: apply the alpha
    substitution only when ``>=50%`` of the token's non-digit-confusable
    characters are letters (i.e. the token reads like a word, not an
    address number).
    """
    def _fix(m: re.Match[str]) -> str:
        tok = m.group(0)
        letters = sum(c.isalpha() for c in tok)
        # At least two letters and more letters than non-letter symbols.
        if letters >= 2 and letters >= len(tok) - letters:
            return tok.translate(_ALPHA_SUBS)
        return tok
    return _ALPHA_TOKEN_RE.sub(_fix, value)
