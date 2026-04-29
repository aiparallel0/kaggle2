"""Arithmetic consensus solver for the SROIE ``total`` field.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: derive the receipt's grand total from arithmetic identities
    over parsed money values rather than picking a money line by
    spatial / keyword cues.

The grand total of a SROIE receipt almost always satisfies one or
more of:

    1. ``total = cash − change``                (cash transactions)
    2. ``total = subtotal + tax (+ service − discount)``

When ≥2 identities agree to ±2¢ the value is committed without
trusting the (often OCR-corrupted) total line itself.  When exactly
one identity fires unambiguously we still commit; otherwise the
caller falls through to the existing keyword-rule chain.

Pure stdlib (no torch / transformers).  Returns ``(line_idx, value)``
where ``line_idx == -1`` indicates a synthesised value not anchored
to any single OCR line — recovers OCR-corrupted total lines that
the scored keyword chain cannot read.
"""
from __future__ import annotations

import re

# Line-type keyword regexes.  Precedence (subtotal > tax > service >
# discount > cash > change > rounding > total) ensures a SUB-TOTAL
# line is never mislabelled as a grand total, and a CASH line is
# never flipped to CHANGE on misordered layouts.
_KW = {
    "subtotal": re.compile(r"\bsub[\s\-]?total\b|\bsubtotal\b", re.IGNORECASE),
    "tax": re.compile(r"\b(?:gst|sst|tax|cukai|vat)\b", re.IGNORECASE),
    "service": re.compile(r"\bservice\s*(?:charge|tax)?\b", re.IGNORECASE),
    "discount": re.compile(
        r"\b(?:discount|disc(?:ount)?|diskaun|rebate)\b", re.IGNORECASE,
    ),
    "cash": re.compile(
        r"\b(?:tunai|cash(?:\s+tendered)?|tendered|bayar(?:an)?|paid)\b",
        re.IGNORECASE,
    ),
    "change": re.compile(
        r"\b(?:change|kembalian|kembali|baki)\b", re.IGNORECASE,
    ),
    "rounding": re.compile(
        r"\b(?:round(?:ing|ed)?|adj(?:ust(?:ment)?)?)\b", re.IGNORECASE,
    ),
    "total": re.compile(
        r"\b(?:grand\s*total|nett?\s*total|total\s*(?:due|amount)?|"
        r"jumlah(?:\s+bersih)?|amount\s+(?:due|payable))\b",
        re.IGNORECASE,
    ),
}
_LABEL_ORDER = (
    "subtotal", "tax", "service", "discount",
    "cash", "change", "rounding", "total",
)
_DISTRACTOR_LABELS = frozenset({
    "subtotal", "tax", "service", "discount", "cash", "change", "rounding",
})
_MONEY_RE = re.compile(r"-?\d{1,3}(?:,\d{3})*\.\d{2}|-?\d+\.\d{2}")
_EPS = 0.02


def _parse_money(text: str) -> float | None:
    """Rightmost parseable money on the line, or ``None``."""
    matches = list(_MONEY_RE.finditer(text or ""))
    if not matches:
        return None
    try:
        return float(matches[-1].group().replace(",", ""))
    except ValueError:
        return None


def _classify(texts: list[str]) -> list[tuple[int, str, float]]:
    """``[(idx, label, value)]`` for every money-bearing line."""
    out: list[tuple[int, str, float]] = []
    for i, t in enumerate(texts):
        v = _parse_money(t)
        if v is None:
            continue
        label = "none"
        for name in _LABEL_ORDER:
            if _KW[name].search(t):
                label = name
                break
        out.append((i, label, v))
    return out


def _vals(c: list[tuple[int, str, float]], label: str) -> list[float]:
    """All values whose line carries ``label``."""
    return [v for _i, lab, v in c if lab == label]


def _identity_cash_change(c: list[tuple[int, str, float]]) -> float | None:
    """Identity 1: ``total = cash − change`` (max-cash − min-change)."""
    cash, change = _vals(c, "cash"), _vals(c, "change")
    if not cash or not change:
        return None
    diff = max(cash) - min(change)
    return diff if diff > 0 else None


def _identity_sub_tax(c: list[tuple[int, str, float]]) -> float | None:
    """Identity 2: ``subtotal + tax + service − discount``."""
    sub = _vals(c, "subtotal")
    if not sub:
        return None
    val = (
        max(sub) + sum(_vals(c, "tax"))
        + sum(_vals(c, "service")) - sum(_vals(c, "discount"))
    )
    return val if val > 0 else None


def _line_with_value(
    c: list[tuple[int, str, float]], target: float,
) -> int:
    """First non-distractor line whose value matches ``target`` to ε."""
    for i, lab, v in c:
        if lab not in _DISTRACTOR_LABELS and abs(v - target) <= _EPS:
            return i
    for i, _lab, v in c:
        if abs(v - target) <= _EPS:
            return i
    return -1


def total_arithmetic_consensus(
    texts: list[str], used: set[int],
) -> tuple[int, str] | None:
    """Solve for ``total`` via arithmetic identities over parsed money.

    Returns ``(line_idx, value_str)`` where ``line_idx >= 0`` is the
    OCR line carrying the consensus value, ``line_idx == -1`` is a
    synthesised value (no line on the receipt carries it verbatim;
    recovers OCR-corrupted total lines), and ``None`` is ambiguous
    or underdetermined (caller should fall through).
    """
    c = [(i, lab, v) for i, lab, v in _classify(texts) if i not in used]
    if not c:
        return None
    cands: list[float] = []
    for fn in (_identity_cash_change, _identity_sub_tax):
        v = fn(c)
        if v is not None and v > 0:
            cands.append(v)
    if not cands:
        return None
    # Two-of-many consensus first.
    for i, v in enumerate(cands):
        for w in cands[i + 1:]:
            if abs(v - w) <= _EPS:
                return _line_with_value(c, v), f"{v:.2f}"
    if len(cands) == 1:
        v = cands[0]
        return _line_with_value(c, v), f"{v:.2f}"
    return None


__all__ = ["total_arithmetic_consensus"]
