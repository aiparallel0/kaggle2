"""Per-field analytical propagation on top of the learned draft picks.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: refines the learned AttentionAssigner's draft field→value dict
    using the same four-step recipe that lifts ``date`` to F1≥0.95 —
    regex/validator, value extractor, runner-up fallback, output
    normaliser.  Per-field fixes: total (SUBTOTAL/``TOTAL:``) →
    money-candidate scoring with ±1-line keyword context; address
    → spatial propagation via rule ``_pick_address``; company →
    validate-and-fallback to topmost non-junk; date → regex-first
    with reading-order fallback.  No torch dep — runs on any
    assigner checkpoint without retraining.
"""
from __future__ import annotations

import re

from models.pipeline_normalize import (
    normalize_address,
    normalize_company,
    normalize_date,
    normalize_total_value,
)
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


def _attn_rank(row: list[float]) -> dict[int, int]:
    """``{region_idx: rank}`` with rank 0 = highest attention."""
    order = sorted(range(len(row)), key=lambda i: -row[i])
    return {i: r for r, i in enumerate(order)}


def _score_money(
    i: int, texts: list[str], bboxes: list[list[float]], rank: dict[int, int],
) -> float:
    """Higher = more likely the real TOTAL line (±1-line keyword context)."""
    nbr = " ".join(texts[max(0, i - 1): i + 2])
    s = 0.0
    if _TOTAL_KW_RE.search(nbr):
        s += 2.0
    if _SUBTOTAL_KW_RE.search(nbr):
        s -= 2.5
    if _TOTAL_NEGATIVE.search(nbr):
        s -= 1.5
    s += 0.5 * (bboxes[i][1] if i < len(bboxes) else 0.0)
    r = rank.get(i, len(rank))
    return s + (1.0 if r == 0 else 0.3 if r <= 2 else 0.0)


def _refine_total(
    learned: str, texts: list[str], bboxes: list[list[float]],
    attn_row: list[float] | None,
) -> str:
    """Score every money-bearing region; pick argmax; fall back to learned."""
    repaired = [repair_money_ocr(t) for t in texts]
    cands = [(i, m.group(0)) for i, t in enumerate(repaired)
             if (m := _MONEY_RE.search(t)) is not None]
    if not cands:
        return learned
    rank = _attn_rank(attn_row) if attn_row else {}
    scored = sorted(
        ((_score_money(i, repaired, bboxes, rank), i, v) for i, v in cands),
        reverse=True,
    )
    best_score, _, best_val = scored[0]
    learned_clean = _strip_currency(learned)
    if best_score <= 0 and _MONEY_RE.fullmatch(learned_clean):
        return learned_clean
    return _strip_currency(best_val)


def _refine_date(learned: str, texts: list[str]) -> str:
    """Keep learned pick if DATE_RE matches; otherwise first regex hit."""
    m = _DATE_RE.search(learned)
    if m is not None:
        return m.group(0)
    picked = extract_date(texts)
    return picked[1] if picked is not None else learned


def _refine_company(
    learned: str, texts: list[str], bboxes: list[list[float]],
) -> str:
    """Validate-and-fallback: HEADER_JUNK / phone / too-short → topmost non-junk."""
    t = learned.strip()
    valid = (not _is_short_junk(t)
             and _HEADER_JUNK.match(t) is None
             and _ADDR_EXCLUDE.search(t) is None)
    if valid:
        return t
    picked = _pick_company(texts, bboxes, used=set())
    return picked[1] if picked is not None else learned


def _y_of(value: str, texts: list[str], bboxes: list[list[float]]) -> float:
    key = value.strip().lower()
    if not key:
        return 0.0
    for i, t in enumerate(texts):
        if key in t.lower():
            return bboxes[i][1] if i < len(bboxes) else 0.0
    return 0.0


def _refine_address(
    learned: str, texts: list[str], bboxes: list[list[float]],
    field_values: dict[str, str],
) -> str:
    """Spatial propagation: union learned multi-line pick with rule pick."""
    total_pick = extract_total(texts, bboxes)
    date_pick = extract_date(texts)
    rule_addr = _pick_address(
        texts, bboxes, used=set(),
        company_y=_y_of(field_values.get("company", ""), texts, bboxes),
        total_y=(bboxes[total_pick[0]][1]
                 if total_pick and total_pick[0] < len(bboxes) else 0.0),
        date_y=(bboxes[date_pick[0]][1]
                if date_pick and date_pick[0] < len(bboxes) else 0.0),
    )

    def _score(s: str) -> tuple[int, int]:
        return (0 if _ADDR_EXCLUDE.search(s) else 1, len(s.strip()))
    return max((learned, rule_addr), key=_score)


def refine_assignments(
    draft: dict[str, str], texts: list[str], bboxes: list[list[float]],
    attn: list[list[float]] | None, fields: list[str],
) -> dict[str, str]:
    """Apply per-field refiner + output normaliser to the learned draft."""
    out = dict(draft)
    by_idx = {f.lower(): i for i, f in enumerate(fields)}
    if "date" in out:
        out["date"] = normalize_date(_refine_date(out["date"], texts))
    if "total" in out:
        row = attn[by_idx["total"]] if attn and "total" in by_idx else None
        out["total"] = normalize_total_value(
            _refine_total(out["total"], texts, bboxes, row),
        )
    if "company" in out:
        out["company"] = normalize_company(
            _refine_company(out["company"], texts, bboxes),
        )
    if "address" in out:
        out["address"] = normalize_address(
            _refine_address(out["address"], texts, bboxes, out),
        )
    return out
