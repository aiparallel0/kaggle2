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
    _TOTAL_STRONG,
    _TOTAL_WEAK,
    repair_money_ocr,
)

_CURRENCY_PREFIX_RE = re.compile(r"^(RM|USD|SGD|MYR|\$)\s*", re.IGNORECASE)
# Address anchor: Malaysian-receipt address-line openers.  The topmost
# region matching this pattern is the first line of the postal address,
# regardless of where the assigner's attention fell.
_ADDR_ANCHOR = re.compile(
    r"\b(NO\.?|LOT|JALAN|JLN|TAMAN|TMN|BANDAR|BDR|PLAZA|"
    r"GROUND|GRD|FLR|FLOOR|KAWASAN|SEKSYEN|BLOCK|BLK|MALL)\b",
    re.IGNORECASE,
)
# Currency-prefix cue on the SAME line as the money value — a weak positive
# because TOTAL lines are the ones most often printed with ``RM``/``MYR``.
_CURRENCY_CUE_RE = re.compile(r"\b(?:RM|MYR|\$)\b", re.IGNORECASE)


def _strip_currency(s: str) -> str:
    return _CURRENCY_PREFIX_RE.sub("", s).strip()


def _attn_rank(row: list[float]) -> dict[int, int]:
    """``{region_idx: rank}`` with rank 0 = highest attention."""
    order = sorted(range(len(row)), key=lambda i: -row[i])
    return {i: r for r, i in enumerate(order)}


def _score_money(
    i: int, texts: list[str], rank: dict[int, int], money_idxs: list[int],
) -> float:
    """Higher = more likely the real TOTAL line.

    Signals (tuned to the kaggle2 SROIE miss table where ~1/3 of losses
    are SUBTOTAL/TAX/CASH/CHANGE picked instead of GRAND TOTAL):

    - ±1-line keyword window: ``_TOTAL_STRONG`` (+4), ``_TOTAL_WEAK``
      w/o SUBTOTAL (+2.5), ``_SUBTOTAL_KW_RE`` (-4), ``_TOTAL_NEGATIVE``
      (-2).  Asymmetric weights prevent a ``SUBTOTAL: RM 38`` line that
      is also near ``TOTAL:`` (two lines down) from winning.
    - ``RM``/``MYR`` cue on the SAME line as the money value (+0.3) —
      grand totals are the ones printed with a currency symbol.
    - Positional: the **last** money line gets +1.5, the 2nd-to-last
      gets +0.5 (this is the strongest single signal on SROIE).
    - Attention rank tie-break: argmax +1.0, top-3 +0.3.

    Note the previous version multiplied ``bboxes[i][1]`` (raw pixel y)
    by 0.5 which produced dominating 500+ scores on tall receipts; the
    pixel-y term is removed and replaced by the relative money-line
    position above.
    """
    nbr = " ".join(texts[max(0, i - 1): i + 2])
    same_line = texts[i] if i < len(texts) else ""
    s = 0.0
    if _TOTAL_STRONG.search(nbr):
        s += 4.0
    elif _TOTAL_WEAK.search(nbr) and not _SUBTOTAL_KW_RE.search(nbr):
        s += 2.5
    if _SUBTOTAL_KW_RE.search(nbr):
        s -= 4.0
    if _TOTAL_NEGATIVE.search(nbr):
        s -= 2.0
    if _CURRENCY_CUE_RE.search(same_line):
        s += 0.3
    if money_idxs:
        if i == money_idxs[-1]:
            s += 1.5
        elif len(money_idxs) >= 2 and i == money_idxs[-2]:
            s += 0.5
    r = rank.get(i, len(rank))
    if r == 0:
        s += 1.0
    elif r <= 2:
        s += 0.3
    return s


def _refine_total(
    learned: str, texts: list[str], bboxes: list[list[float]],
    attn_row: list[float] | None,
) -> str:
    """Score every money-bearing region; pick argmax; fall back to learned.

    *Number fields should be near-100%* — unlike address, a total is a
    single value with a tight regex. We therefore prefer the scored
    pick over the learned argmax whenever the scorer finds any strong
    positive evidence (``score > 0``); we only keep the learned value
    when every money candidate has negative score (subtotal / tax /
    cash-only receipt) AND the learned pick is a well-formed money
    string.
    """
    repaired = [repair_money_ocr(t) for t in texts]
    money_idxs = [i for i, t in enumerate(repaired) if _MONEY_RE.search(t)]
    if not money_idxs:
        return learned
    rank = _attn_rank(attn_row) if attn_row else {}
    scored = sorted(
        ((_score_money(i, repaired, rank, money_idxs), i,
          _MONEY_RE.search(repaired[i]).group(0))  # type: ignore[union-attr]
         for i in money_idxs),
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


def _is_addr_boundary(t: str) -> bool:
    """Address span terminator: money / date / phone-or-tax-id / header junk."""
    return bool(_MONEY_RE.search(t) or _DATE_RE.search(t)
                or _ADDR_EXCLUDE.search(t) or _HEADER_JUNK.match(t))


def _address_span(
    texts: list[str], bboxes: list[list[float]],
) -> str:
    """Greedy spatial span anchored on the topmost address-keyword region.

    Fixes the #1 failure mode observed in the miss table: the assigner
    picks only line 1-2 of a 4-5-line address because every subsequent
    line's attention falls below ``_MULTI_LINE_FRACTION * max``.  Here
    we IGNORE attention entirely for address and instead:

    1. Find the topmost region whose text matches ``_ADDR_ANCHOR``
       (NO./LOT/JALAN/TAMAN/BANDAR/...) — this is always line 1 of the
       address on Malaysian receipts.
    2. Walk downward in y-order and append every non-junk region until
       hitting a boundary (money line, date line, phone/tax-ID line, or
       header junk).  Small fillers (``_is_short_junk``) are skipped
       but don't break the span.
    3. Return the concatenated text.  Empty string when no anchor is
       found (caller falls back to the learned pick).
    """
    n = len(texts)
    if n == 0:
        return ""
    y_order = sorted(range(n), key=lambda j: bboxes[j][1] if j < len(bboxes) else 0.0)
    # Find topmost anchor in y-order that isn't a header/money/date/phone.
    anchor_pos: int | None = None
    for pos, j in enumerate(y_order):
        t = texts[j].strip()
        if not t or _is_short_junk(t) or _is_addr_boundary(t):
            continue
        if _ADDR_ANCHOR.search(t):
            anchor_pos = pos
            break
    if anchor_pos is None:
        return ""
    picked: list[int] = [y_order[anchor_pos]]
    for j in y_order[anchor_pos + 1:]:
        t = texts[j].strip()
        if not t or _is_short_junk(t):
            continue
        if _is_addr_boundary(t):
            break
        picked.append(j)
    return " ".join(texts[j].strip() for j in picked if texts[j].strip())


def _refine_address(
    learned: str, texts: list[str], bboxes: list[list[float]],
    field_values: dict[str, str],
) -> str:
    """Prefer the longest junk-free candidate among (learned, rule, span).

    The miss table shows ~60% of address losses are pure prefix-of-GT
    (under-picked).  The greedy ``_address_span`` consistently beats
    both the learned attention pick and the rule-based ``_pick_address``
    on multi-line addresses because it anchors on address keywords
    instead of company-y / total-y boundaries that may be missing or
    wrong on receipts with unusual layouts.
    """
    span = _address_span(texts, bboxes)
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
    return max((learned, rule_addr, span), key=_score)


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
