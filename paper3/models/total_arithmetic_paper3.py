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

# FOCUS-Σ subset-sum bounds.  SROIE receipts have ≤30 money-bearing
# lines; the largest plausible per-item amount is RM 5000 (500,000¢).
# A per-item filter keeps a single noise line ("SOMETHING 4838.20"
# parsed off a phone number / registration ID) from inflating the
# DP bound past tractability — the DP table is sized to the *sum
# after filtering*, so noise lines drop out cleanly.
_SUBSET_SUM_MAX_ITEMS = 30
_SUBSET_SUM_MAX_ITEM_CENTS = 500_000
_SUBSET_SUM_MAX_SUM_CENTS = 1_000_000


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


def subset_sum_target_cents(
    c: list[tuple[int, str, float]],
) -> frozenset[int]:
    """FOCUS-Σ Identity 3: ``total = Σ(items) + Σ(tax) + Σ(service) − Σ(discount)``.

    Returns the cent-integer set of values reachable by summing a
    non-empty subset of the receipt's non-distractor money-bearing
    lines (``label == "none"``, ``v > 0``) and adding the augmented
    tax term ``Σ(tax) + Σ(service) − Σ(discount)``.

    Rationale.  Identities 1 and 2 (cash−change, subtotal+tax+…) are
    keyword-anchored: when SROIE OCR loses ``SUBTOTAL`` / ``CASH`` /
    ``CHANGE`` they go silent and the consensus pass falls through to
    keyword-free scoring.  The grand-total of every SROIE receipt is
    structurally ``Σ(items) + tax_aug`` regardless of which summary
    keywords the printer chose to emit; Identity 3 makes that
    structural fact a witness without depending on any keyword.

    Computational shape.  SROIE receipts have ≤30 money lines and
    totals < RM 2000.  Subset-sum DP over a bounded boolean array is
    O(N × max_sum_cents) — milliseconds per receipt.  When the receipt
    exceeds the bounds (long invoices, foreign-currency totals) the
    function returns the empty set: callers fall through to the
    legacy 2-identity consensus rather than risk an unbounded DP.

    Strictly additive (FOCUS-T-compatible).  Combined with
    :func:`_identity_cash_change` and :func:`_identity_sub_tax` this
    raises the witness ceiling from 2 to 3 — a count of 3 is
    essentially proof on SROIE; existing thresholds at counts 1 and 2
    keep their semantics.
    """
    items = [v for _i, lab, v in c if lab == "none" and v > 0.0]
    if not items or len(items) > _SUBSET_SUM_MAX_ITEMS:
        return frozenset()
    tax_aug = (
        sum(_vals(c, "tax")) + sum(_vals(c, "service"))
        - sum(_vals(c, "discount"))
    )
    # Drop per-line outliers (>RM 5000) before sizing the DP — a stray
    # phone number / registration ID parsed as money would otherwise push
    # the bound past tractability and make the function return empty
    # exactly when the receipt has the most noise.
    item_cents = [
        int(round(v * 100)) for v in items
        if 0 < v * 100 <= _SUBSET_SUM_MAX_ITEM_CENTS
    ]
    if not item_cents:
        return frozenset()
    tax_cents = int(round(tax_aug * 100))
    bound = sum(item_cents)
    if bound + max(0, tax_cents) > _SUBSET_SUM_MAX_SUM_CENTS:
        return frozenset()
    # Subset-sum DP tracking minimum cardinality so we can require
    # ``cardinality ≥ 2`` and exclude trivial singleton-self witnesses
    # (a noise money line ``4838.20`` parsed off a phone number would
    # otherwise be its own witness via the size-1 subset {4838.20}).
    inf = len(item_cents) + 1
    card = [inf] * (bound + 1)
    card[0] = 0
    for v in item_cents:
        for s in range(bound, v - 1, -1):
            new = card[s - v] + 1
            if new < card[s]:
                card[s] = new
    # Witness requires cardinality ≥ 2 — either two items alone, or one
    # item plus a non-zero tax/service/discount augmentation (which
    # itself is a second "summand" structurally).
    if tax_cents != 0:
        return frozenset(
            s + tax_cents for s in range(1, bound + 1)
            if 1 <= card[s] <= len(item_cents)
            and 0 <= s + tax_cents <= _SUBSET_SUM_MAX_SUM_CENTS
        )
    return frozenset(
        s for s in range(1, bound + 1) if 2 <= card[s] <= len(item_cents)
    )


def total_witness_count(
    value: float, classified: list[tuple[int, str, float]],
) -> int:
    """3-identity FOCUS-Σ witness count for a candidate total ``value``.

    Returns how many of {I₁: cash−change, I₂: subtotal+tax+svc−disc,
    I₃: Σ(items)+tax_aug subset-sum} ``value`` satisfies to ±2¢.  The
    inference-time companion of the training-time per-line prior
    column :func:`models.focus_priors.arithmetic_witnesses_v4`.
    """
    count = 0
    cc = _identity_cash_change(classified)
    if cc is not None and abs(cc - value) <= _EPS:
        count += 1
    st = _identity_sub_tax(classified)
    if st is not None and abs(st - value) <= _EPS:
        count += 1
    if int(round(value * 100)) in subset_sum_target_cents(classified):
        count += 1
    return count


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
    p_totals: list[float] | None = None,
    totals_zone_floor: float = 0.0,
) -> tuple[int, str] | None:
    """Solve for ``total`` via arithmetic identities over parsed money.

    Returns ``(line_idx, value_str)`` where ``line_idx >= 0`` is the
    OCR line carrying the consensus value, ``line_idx == -1`` is a
    synthesised value (no line on the receipt carries it verbatim;
    recovers OCR-corrupted total lines), and ``None`` is ambiguous
    or underdetermined (caller should fall through).

    When a relational ``ZonePosterior`` is supplied via ``p_totals``
    (length-aligned to ``texts``), candidate-money lines whose
    ``p_total`` is below ``totals_zone_floor`` are filtered out before
    enumerating identities — phone numbers, registration suffixes and
    other header-zone numerics that the money regex would otherwise
    accept never reach the arithmetic enumeration.  Defaults preserve
    the legacy bit-exact behaviour (no zone gate) when ``p_totals`` is
    ``None``.
    """
    classified = _classify(texts)
    if p_totals is not None:
        classified = [
            (i, lab, v) for i, lab, v in classified
            if i < len(p_totals) and p_totals[i] >= totals_zone_floor
        ]
    c = [(i, lab, v) for i, lab, v in classified if i not in used]
    if not c:
        return None
    cands: list[float] = []
    for fn in (_identity_cash_change, _identity_sub_tax):
        v = fn(c)
        if v is not None and v > 0:
            cands.append(v)
    # FOCUS-Σ note: when I₁/I₂ are both silent the consensus stays None
    # by design — Identity 3 (items-subset-sum) alone is underdetermined
    # at this layer (multiple item-line values are reachable as their
    # own subset-sum and there is no in-band tie-break).  I₃ contributes
    # via the per-line witness boost in :func:`models.consensus._score_money`
    # where the keyword/positional/attention signals break ties.
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


__all__ = [
    "subset_sum_target_cents",
    "total_arithmetic_consensus",
    "total_witness_count",
]
