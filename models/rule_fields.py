"""Per-field extractors used by the rule-based KIE baseline."""
from __future__ import annotations

import re

from models.rule_regex import (
    _ADDR_EXCLUDE,
    _DATE_RE,
    _HEADER_JUNK,
    _MONEY_RE,
    _TOTAL_NEGATIVE,
    _TOTAL_STRONG,
    _TOTAL_WEAK,
)


def _is_short_junk(text: str) -> bool:
    """True if ``text`` is too short/digit-heavy/punctuation-only to be a
    plausible company name or address line."""
    s = text.strip()
    if len(s) < 3:
        return True
    alpha = sum(c.isalpha() for c in s)
    return alpha < 3


def extract_total(
    region_texts: list[str], bbox_list: list[list[float]],
) -> tuple[int, str] | None:
    """Pick the best TOTAL region → (index, money-substring) or None.

    Ranking (higher = better): 4) GRAND TOTAL + money, 3) TOTAL/AMOUNT + money,
    2) money in bottom half, 1) money in top half, 0) money + negative word.
    Ties on score broken by bottom-most y-position.
    """
    candidates: list[tuple[int, int, float, str]] = []
    for i, txt in enumerate(region_texts):
        m = _MONEY_RE.search(txt)
        if not m:
            continue
        t = txt.strip()
        neg = bool(_TOTAL_NEGATIVE.search(t))
        strong = bool(_TOTAL_STRONG.search(t))
        weak = bool(_TOTAL_WEAK.search(t))
        y = bbox_list[i][3] if i < len(bbox_list) else 0.0
        if strong and not neg:
            score = 4
        elif weak and not neg:
            score = 3
        elif not neg:
            score = 2 if y >= 0.5 else 1
        else:
            score = 0
        candidates.append((score, i, y, m.group(0).strip()))
    if not candidates:
        return None
    best = max(candidates, key=lambda c: (c[0], c[2]))
    if best[0] == 0:
        # Every candidate is negated (change / rounding); keep the first one.
        best = max(candidates, key=lambda c: c[2])
    # Strip currency prefix so the value matches SROIE GT ("12.30", not "RM12.30").
    value = re.sub(r"^(RM|USD|SGD|MYR|\$)\s*", "", best[3], flags=re.IGNORECASE)
    return best[1], value


def extract_date(region_texts: list[str]) -> tuple[int, str] | None:
    """First region matching DATE_RE → (index, date-substring)."""
    for i, txt in enumerate(region_texts):
        m = _DATE_RE.search(txt)
        if m:
            return i, m.group(0)
    return None


def pick_company(
    region_texts: list[str], bbox_list: list[list[float]], used: set[int],
) -> tuple[int, str] | None:
    """Top-most region that is not junk, a header, a date, or money."""
    order = sorted(
        [i for i in range(len(region_texts)) if i not in used],
        key=lambda i: bbox_list[i][1] if i < len(bbox_list) else 0.0,
    )
    for i in order:
        t = region_texts[i].strip()
        if _is_short_junk(t) or _HEADER_JUNK.match(t):
            continue
        if _DATE_RE.search(t) or _MONEY_RE.search(t) or _ADDR_EXCLUDE.search(t):
            continue
        return i, t
    if order:
        return order[0], region_texts[order[0]].strip()
    return None


def pick_address(
    region_texts: list[str], bbox_list: list[list[float]],
    used: set[int], company_y: float, total_y: float, date_y: float,
    max_lines: int = 6,
) -> str:
    """Concatenate address lines between the company and the first money /
    date region (address is always above money on SROIE)."""
    lower_bound = (
        min(x for x in (total_y, date_y) if x > 0)
        if (total_y > 0 or date_y > 0) else 1.0
    )
    candidates: list[tuple[float, int, str]] = []
    for i in range(len(region_texts)):
        if i in used:
            continue
        y = bbox_list[i][1] if i < len(bbox_list) else 0.0
        if y <= company_y or y >= lower_bound:
            continue
        t = region_texts[i].strip()
        if _is_short_junk(t):
            continue
        if _DATE_RE.search(t) or _MONEY_RE.search(t):
            continue
        if _ADDR_EXCLUDE.search(t) or _HEADER_JUNK.match(t):
            continue
        candidates.append((y, i, t))
    candidates.sort()
    lines = [t for _, _, t in candidates[:max_lines]]
    if lines:
        return " ".join(lines)
    # Looser fallback when spatial filter is empty.
    by_y = sorted(
        [(i, bbox_list[i][1] if i < len(bbox_list) else 0.0)
         for i in range(len(region_texts)) if i not in used],
        key=lambda x: x[1],
    )
    fallback: list[str] = []
    for i, _ in by_y:
        t = region_texts[i].strip()
        if _MONEY_RE.search(t) or _DATE_RE.search(t):
            continue
        if _ADDR_EXCLUDE.search(t) or _HEADER_JUNK.match(t):
            continue
        if _is_short_junk(t):
            continue
        fallback.append(t)
        if len(fallback) >= 4:
            break
    return " ".join(fallback)
