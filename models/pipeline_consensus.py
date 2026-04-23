"""Per-field analytical propagation on top of the learned draft picks.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: refines the learned AttentionAssigner's draft field→value dict
    with a per-field rule tailored to that field's failure mode.  No
    torch dep — runs on any checkpoint without retraining.

Per-field failure modes (SROIE 63-image test):
  * total    — argmax lands on ``TOTAL:`` or SUBTOTAL.  Fix: money
               candidate scoring with ±1-line TOTAL-keyword context,
               penalty on SUBTOTAL / CASH-TENDERED / change lines.
  * address  — under-pick (attn spread) or over-pick (junk in 0.5×max
               band).  Fix: spatial propagation between company_y and
               first money/date line, rejecting ADDR_EXCLUDE lines.
  * company  — ``TAX INVOICE`` / ``CASH BILL`` argmaxed over the real
               name.  Fix: HEADER_JUNK reject + topmost non-junk fallback.
  * date     — the learned pick rarely misses; retained unless its
               text carries no DATE_RE match.
"""
from __future__ import annotations

import re

from models.rule_fields import (
    _SUBTOTAL_KW_RE,
    _TOTAL_KW_RE,
    _is_short_junk,
    _pick_address,
    _pick_company,
    extract_date,
    extract_total,
)
from models.rule_regex import (
    _ADDR_EXCLUDE,
    _DATE_RE,
    _HEADER_JUNK,
    _MONEY_RE,
    _TOTAL_NEGATIVE,
    repair_money_ocr,
)

_CURRENCY_PREFIX_RE = re.compile(r"^(RM|USD|SGD|MYR|\$)\s*", re.IGNORECASE)


def _strip_currency(s: str) -> str:
    return _CURRENCY_PREFIX_RE.sub("", s).strip()


def _attn_rank(attn_row: list[float]) -> dict[int, int]:
    """``{region_idx: rank}`` with rank 0 = highest attention."""
    order = sorted(range(len(attn_row)), key=lambda i: -attn_row[i])
    return {i: r for r, i in enumerate(order)}


def _score_money_region(
    i: int, texts: list[str], bboxes: list[list[float]],
    rank: dict[int, int],
) -> float:
    """Higher = more likely the real TOTAL line (±1-line keyword context)."""
    nbr = " ".join(texts[max(0, i - 1): i + 2])
    score = 0.0
    if _TOTAL_KW_RE.search(nbr):
        score += 2.0
    if _SUBTOTAL_KW_RE.search(nbr):
        score -= 2.5
    if _TOTAL_NEGATIVE.search(nbr):
        score -= 1.5
    score += 0.5 * (bboxes[i][1] if i < len(bboxes) else 0.0)
    r = rank.get(i, len(rank))
    score += 1.0 if r == 0 else (0.3 if r <= 2 else 0.0)
    return score


def _refine_total(
    learned_value: str, texts: list[str], bboxes: list[list[float]],
    attn_row: list[float] | None,
) -> str:
    """Score every money-bearing region; pick argmax; fall back to learned."""
    repaired = [repair_money_ocr(t) for t in texts]
    cands = [(i, m.group(0)) for i, t in enumerate(repaired)
             if (m := _MONEY_RE.search(t)) is not None]
    if not cands:
        return learned_value
    rank = _attn_rank(attn_row) if attn_row else {}
    scored = [(_score_money_region(i, repaired, bboxes, rank), i, v)
              for i, v in cands]
    scored.sort(reverse=True)
    best_score, _, best_val = scored[0]
    learned_clean = _strip_currency(learned_value)
    if best_score <= 0 and _MONEY_RE.fullmatch(learned_clean):
        return learned_clean
    return _strip_currency(best_val)


def _refine_date(learned_value: str, texts: list[str]) -> str:
    """Keep learned pick if DATE_RE matches; otherwise first regex hit."""
    m = _DATE_RE.search(learned_value)
    if m is not None:
        return m.group(0)
    picked = extract_date(texts)
    return picked[1] if picked is not None else learned_value


def _refine_company(
    learned_value: str, texts: list[str], bboxes: list[list[float]],
) -> str:
    """If learned pick is HEADER_JUNK or too short, take topmost non-junk."""
    t = learned_value.strip()
    ok = (
        not _is_short_junk(t)
        and _HEADER_JUNK.match(t) is None
        and _ADDR_EXCLUDE.search(t) is None
    )
    if ok:
        return t
    picked = _pick_company(texts, bboxes, used=set())
    return picked[1] if picked is not None else learned_value


def _y_of_value(value: str, texts: list[str], bboxes: list[list[float]]) -> float:
    """Find the y1 of the first region whose lower-cased text contains ``value``."""
    v = value.strip().lower()
    if not v:
        return 0.0
    for i, t in enumerate(texts):
        if v in t.lower():
            return bboxes[i][1] if i < len(bboxes) else 0.0
    return 0.0


def _refine_address(
    learned_value: str, texts: list[str], bboxes: list[list[float]],
    field_values: dict[str, str],
) -> str:
    """Spatial propagation: rule pick between company_y and first money/date."""
    company_y = _y_of_value(field_values.get("company", ""), texts, bboxes)
    total_pick = extract_total(texts, bboxes)
    total_y = (
        bboxes[total_pick[0]][1]
        if total_pick and total_pick[0] < len(bboxes) else 0.0
    )
    date_pick = extract_date(texts)
    date_y = (
        bboxes[date_pick[0]][1]
        if date_pick and date_pick[0] < len(bboxes) else 0.0
    )
    rule_addr = _pick_address(
        texts, bboxes, used=set(),
        company_y=company_y, total_y=total_y, date_y=date_y,
    )
    # Prefer the longer, junk-free string — learned multi-line sometimes
    # drops the postcode; rule propagation sometimes includes a phone line.
    def _score(s: str) -> tuple[int, int]:
        junk = 1 if _ADDR_EXCLUDE.search(s) else 0
        return (-junk, len(s.strip()))
    return max((learned_value, rule_addr), key=_score)


def refine_assignments(
    draft: dict[str, str], texts: list[str], bboxes: list[list[float]],
    attn: list[list[float]] | None, fields: list[str],
) -> dict[str, str]:
    """Apply per-field analytical propagation on top of the learned draft.

    ``attn[field_idx][region_idx]`` is the learned cross-attention row per
    field (or ``None`` when unavailable — e.g. empty-detect fallback).
    Returns a new dict — the input is not mutated.
    """
    out = dict(draft)
    by_idx = {f.lower(): idx for idx, f in enumerate(fields)}
    if "date" in out:
        out["date"] = _refine_date(out["date"], texts)
    if "total" in out:
        row = attn[by_idx["total"]] if attn and "total" in by_idx else None
        out["total"] = _refine_total(out["total"], texts, bboxes, row)
    if "company" in out:
        out["company"] = _refine_company(out["company"], texts, bboxes)
    if "address" in out:
        out["address"] = _refine_address(out["address"], texts, bboxes, out)
    return out
