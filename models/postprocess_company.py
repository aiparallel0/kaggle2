"""FOCUS-C company normaliser + greedy span assembler (PR FOCUS-COMPANY).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: mirrors :mod:`models.postprocess_address` for FOCUS-A.  Exposes
    :func:`normalize_company_focus` (symmetric pred/GT casefold +
    punctuation strip composed with legacy ``normalize_company`` inside
    :mod:`models.normalize_bundle`) and :func:`_company_span` (greedy
    anchor + 0..2 forward-extension lines for ``SDN BHD`` / ``(REG-NO)``
    suffixes, skipping ``TAX INVOICE`` / ``CASH BILL`` boilerplate).
    Driven by the FOCUS-C argmax in
    :func:`models.focus_pipeline._assign_learned_with_attn`.
"""
from __future__ import annotations

import re

from models.consensus import _median_line_height

__all__ = [
    "_COMPANY_ANCHOR",
    "_COMPANY_BOILERPLATE_KW",
    "_COMPANY_SUFFIX",
    "_company_span",
    "normalize_company_focus",
]

# FOCUS-C anchor regex.  Matches a "looks-like-a-merchant-trade-name"
# line: at least two alphabetic tokens AND a majority-uppercase ratio
# (Malaysian/Singaporean SROIE receipts almost always print the company
# name in all-caps).  Pure-numeric or short single-word lines do not
# match.
_COMPANY_ANCHOR = re.compile(r"[A-Z][A-Z&'\-]+\s+[A-Z][A-Z&'\-]+")

# Whole-line boilerplate keywords above the merchant trade name; the
# anchor search must SKIP these.  Combined with priors_v4's learned
# ``is_company_boilerplate`` column inside :func:`_company_span`.
_COMPANY_BOILERPLATE_KW = re.compile(
    r"\b(?:TAX\s+INVOICE|CASH\s+BILL|RECEIPT|OFFICIAL\s+RECEIPT|"
    r"CUSTOMER\s+COPY|MERCHANT\s+COPY|SIMPLIFIED\s+TAX\s+INVOICE|"
    r"CREDIT\s+NOTE|DELIVERY\s+ORDER|INVOICE)\b",
    re.IGNORECASE,
)

# Forward-extension suffix regex: the trailing legal-entity /
# registration-number lines that SROIE OCR splits onto a row of their
# own (``"SDN BHD"`` / ``"(M) SDN BHD"`` / ``"(123456-A)"``).
_COMPANY_SUFFIX = re.compile(
    r"^\s*(?:"
    r"\(?\s*M\s*\)?\s*SDN[\s\-]?BHD|SDN[\s\-]?BHD|BHD|BERHAD"
    r"|PTE[\s\-]?LTD|PTY[\s\-]?LTD|LLC|LLP|GMBH|LIMITED|LTD\.?"
    r"|ENTERPRISE(?:S)?|TRADING|MARKETING|HOLDINGS"
    r"|\(\d+[\-\dA-Z]*\)|\d{5,9}[\s\-]?[A-Z]?|CO\.?"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)
_MONEY_RE = re.compile(r"\d{1,3}(?:,\d{3})*\.\d{2}\b")
_DATE_RE = re.compile(r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b")
_PHONE_RE = re.compile(r"(?:TEL|FAX|PHONE)[\s:.\-]*\d", re.IGNORECASE)

# Symmetric normaliser bits — mirror :mod:`models.postprocess_address`.
_STRIP_PUNCT = ",.:;"
_DIGIT_RE = re.compile(r"\d")
_MULTI_WS_RE = re.compile(r"\s+")


def _strip_token_punct(token: str) -> str:
    """Strip ``,.:;`` from a token unless it carries any digit."""
    if _DIGIT_RE.search(token):
        return token
    return token.translate(str.maketrans("", "", _STRIP_PUNCT))


def normalize_company_focus(value: str) -> str:
    """Symmetric company normaliser — line order preserved, casefold output.

    Mirrors :func:`models.postprocess_address.normalize_address_focus`
    so pred ``"UNIHAKKA … SDN. BHD."`` and GT ``"UNIHAKKA … SDN BHD"``
    reduce to the same casefolded token set.  No-op on empty input.
    """
    if not value:
        return ""
    lines = [ln.strip() for ln in value.splitlines() if ln.strip()]
    joined = " ".join(lines) if lines else value
    collapsed = _MULTI_WS_RE.sub(" ", joined).strip()
    tokens = [_strip_token_punct(t) for t in collapsed.split(" ")]
    tokens = [t for t in tokens if t]
    return " ".join(t.casefold() for t in tokens)


def _is_company_boilerplate_line(text: str, prior_hit: bool) -> bool:
    """Whole-line boilerplate (``TAX INVOICE`` / ``CASH BILL`` / …) OR
    a positive ``priors_v4[is_company_boilerplate]`` from the upstream
    network — either signal disqualifies the line as a company anchor
    or forward-extension target.
    """
    return prior_hit or bool(_COMPANY_BOILERPLATE_KW.search(text))


def _is_money_or_date_line(text: str) -> bool:
    """True iff ``text`` carries a money / date / phone token."""
    return bool(
        _MONEY_RE.search(text)
        or _DATE_RE.search(text)
        or _PHONE_RE.search(text),
    )


def _is_anchor_candidate(text: str) -> bool:
    """≥2 alpha tokens, mostly-upper-case (matches :data:`_COMPANY_ANCHOR`)."""
    s = text.strip()
    if not s or len(s) < 2:
        return False
    return bool(_COMPANY_ANCHOR.search(s))


def _company_span(
    texts: list[str], bboxes: list[list[float]],
    boilerplate_priors: list[bool],
    anchor_idx: int | None = None, gap_mult: float = 2.0,
) -> tuple[list[int], str]:
    """Greedy top-N spatial span: anchor + 0..2 forward-extension lines.

    Anchor: ``anchor_idx`` (FOCUS-C argmax) wins iff it passes the
    boilerplate/money/date/phone gates AND matches
    :func:`_is_anchor_candidate`; otherwise fall back to the topmost
    line (lowest ``y1``) that passes those gates.  Forward-extends at
    most two subsequent lines while the line matches
    :data:`_COMPANY_SUFFIX`, the vertical gap to the previous line is
    ``<= gap_mult * median_line_height`` (mirrors
    :func:`models.consensus.enforce_address_contiguity`), and the line
    is not flagged by ``priors_v4[is_company_boilerplate]``.  Returns
    ``(picks, value)`` in top→bottom order; ``([], "")`` on no anchor.
    """
    n = len(texts)
    if n == 0 or len(bboxes) != n or len(boilerplate_priors) != n:
        return [], ""
    stripped = [t.strip() for t in texts]

    def _is_skippable(i: int) -> bool:
        return (
            _is_company_boilerplate_line(stripped[i], boilerplate_priors[i])
            or _is_money_or_date_line(stripped[i])
        )

    # Resolve the anchor.  Prefer the FOCUS-C argmax, validate, fall back.
    chosen: int | None = None
    if (
        anchor_idx is not None and 0 <= anchor_idx < n
        and stripped[anchor_idx]
        and not _is_skippable(anchor_idx)
        and _is_anchor_candidate(stripped[anchor_idx])
    ):
        chosen = anchor_idx
    if chosen is None:
        order = sorted(
            range(n),
            key=lambda i: bboxes[i][1] if len(bboxes[i]) >= 2 else 0.0,
        )
        for i in order:
            if (
                stripped[i]
                and not _is_skippable(i)
                and _is_anchor_candidate(stripped[i])
            ):
                chosen = i
                break
    if chosen is None:
        return [], ""

    picks = [chosen]
    mh = _median_line_height(bboxes)
    # Order remaining indices below the anchor by ``y1`` so the
    # extension walks down in reading order.
    below = sorted(
        (i for i in range(n) if i != chosen
         and len(bboxes[i]) >= 4
         and bboxes[i][1] >= bboxes[chosen][1]),
        key=lambda i: bboxes[i][1],
    )
    extended = 0
    for i in below:
        if extended >= 2:
            break
        prev = picks[-1]
        if len(bboxes[prev]) < 4:
            break
        if mh > 0.0:
            gap = bboxes[i][1] - bboxes[prev][3]
            if gap > gap_mult * mh:
                break
        if boilerplate_priors[i]:
            continue
        if _is_money_or_date_line(stripped[i]):
            break
        if not _COMPANY_SUFFIX.match(stripped[i]):
            # Non-suffix non-skippable line below the anchor — stop the
            # forward walk so we don't drag in unrelated text.
            break
        picks.append(i)
        extended += 1

    value = " ".join(stripped[i] for i in picks if stripped[i])
    return picks, value
