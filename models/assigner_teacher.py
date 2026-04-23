"""Hard-negative region sets + rule-based teacher scores for the assigner.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: implements strategies B (hard-negative loss term) and C (rule-based
    KD) from the assigner plan.  ``hard_negatives`` returns per-field
    distractor region sets derived from the same regexes ``pipeline_*``
    uses at inference — ``_SUBTOTAL_KW_RE`` / ``_TOTAL_NEGATIVE`` for
    total, ``_ADDR_EXCLUDE`` / ``_HEADER_JUNK`` for address/company.
    ``teacher_distribution`` produces a per-field softmax over regions
    from the rule-based scorers ``_score_money`` (total) and a spatial
    prior (address) so the KD term has supervision even on receipts
    whose GT does not disambiguate SUBTOTAL from TOTAL.

    Pure-Python (no torch); the trainer materialises tensors on demand.
"""
from __future__ import annotations

import math

from models.attention_priors import _MONEY_RE
from models.pipeline_consensus import _score_money
from models.rule_regex import (
    _ADDR_EXCLUDE,
    _DATE_RE,
    _HEADER_JUNK,
    _TOTAL_NEGATIVE,
)

# Fields the plan's hard-neg / KD terms target.  ``company`` / ``date``
# are already near ceiling (F1 ≥ 0.90 on the live miss table) so
# introducing extra loss pressure there would cost more than it gains.
_B_FIELDS = ("total", "address", "company")

# Weight of the rule-based teacher distribution in the KD loss (strategy C).
# 0.1 is low enough that the hard ground-truth pos-mass term still
# dominates when GT and rule agree; high enough that the SUBTOTAL vs.
# TOTAL decision gets positive supervision on receipts where only the
# rule scorer can disambiguate.
KD_TEMPERATURE = 2.0


def hard_negatives(
    texts: list[str], positives: dict[int, list[int]],
    field_to_idx: dict[str, int],
) -> dict[int, list[int]]:
    """Per-field hard-negative region indices (strategy B).

    A negative for field ``f`` is any region that:
      * is not a positive for ``f``, **and**
      * matches a canonical distractor regex for ``f``.

    Returned as ``{f_idx: [region_indices]}`` — empty lists are dropped
    so the loss term only fires when the receipt actually contains a
    plausible confuser.  Callers must skip fields that are not in
    :data:`_B_FIELDS` (company/total/address) because date has no clean
    distractor set on SROIE.
    """
    out: dict[int, list[int]] = {}
    for name in _B_FIELDS:
        f_idx = field_to_idx.get(name)
        if f_idx is None:
            continue
        pos = set(positives.get(f_idx, ()))
        negs: list[int] = []
        for i, t in enumerate(texts):
            if i in pos:
                continue
            s = t.strip()
            if not s:
                continue
            if name == "total":
                # Any line matching SUBTOTAL / CASH / CHANGE / TAX /
                # etc. *or* any other money-bearing line on the receipt.
                if _TOTAL_NEGATIVE.search(s) or (
                    _MONEY_RE.search(s) and not _contains_any_pos_text(s, texts, pos)
                ):
                    negs.append(i)
            elif name == "address" and (
                _ADDR_EXCLUDE.search(s) or _HEADER_JUNK.match(s)
                or _DATE_RE.search(s)
            ):
                # Phone / tax-id / header / date / company headers: anything
                # that is *not* address but could be latched onto.
                negs.append(i)
            elif name == "company" and (
                _ADDR_EXCLUDE.search(s) or _HEADER_JUNK.match(s)
            ):
                # Addresses, phone / tax-id lines, generic header junk are
                # the top confusers for company picks.
                negs.append(i)
        if negs:
            out[f_idx] = negs
    return out


def _contains_any_pos_text(s: str, texts: list[str], pos: set[int]) -> bool:
    """True when ``s`` literally contains any positive region's text —
    prevents flagging a merged OCR line that already covers the GT."""
    return any(
        0 <= p < len(texts) and texts[p] and texts[p].strip() in s
        for p in pos
    )


def teacher_distribution(
    texts: list[str], field_to_idx: dict[str, int],
) -> dict[int, list[float]]:
    """Rule-based teacher distribution over regions, per field (strategy C).

    For ``total``: softmax over :func:`models.pipeline_consensus._score_money`
    applied to every money-bearing region, zeros elsewhere.  For
    ``address``: uniform over regions whose text matches neither a
    distractor nor a header — a weak positional prior, but still the
    best cheap supervision available without re-running the span
    builder for every training step.

    Returns ``{f_idx: probs}`` where ``probs`` sums to 1 over the N
    regions of this receipt.  Missing fields are simply absent from the
    dict — caller must fall back to the pos-mass target.
    """
    n = len(texts)
    out: dict[int, list[float]] = {}
    # --- total ---------------------------------------------------------
    t_idx = field_to_idx.get("total")
    if t_idx is not None and n > 0:
        money_idxs = [i for i, t in enumerate(texts) if _MONEY_RE.search(t)]
        if money_idxs:
            scores = [-1e9] * n
            rank: dict[int, int] = {}  # no attention yet; empty rank
            for i in money_idxs:
                scores[i] = _score_money(i, texts, rank, money_idxs)
            probs = _softmax_masked(scores, KD_TEMPERATURE)
            if sum(probs) > 0:
                out[t_idx] = probs
    # --- address -------------------------------------------------------
    a_idx = field_to_idx.get("address")
    if a_idx is not None and n > 0:
        valid = [
            i for i, t in enumerate(texts)
            if t.strip() and not _ADDR_EXCLUDE.search(t)
            and not _HEADER_JUNK.match(t.strip())
            and not _DATE_RE.search(t) and not _MONEY_RE.search(t)
        ]
        if valid:
            p = [0.0] * n
            w = 1.0 / len(valid)
            for i in valid:
                p[i] = w
            out[a_idx] = p
    return out


def _softmax_masked(scores: list[float], temperature: float) -> list[float]:
    """Softmax with −1e9 treated as hard-zero output; numerically stable."""
    if not scores:
        return []
    m = max(scores)
    if m <= -1e8:
        return [0.0] * len(scores)
    exps = [math.exp((s - m) / temperature) if s > -1e8 else 0.0 for s in scores]
    z = sum(exps)
    if z <= 0:
        return [0.0] * len(scores)
    return [e / z for e in exps]
