"""Per-field extractor functions for the rule-based KIE baseline.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: implements the spatial + regex heuristics for date/total/company/address
    that collectively lift rule-based F1 from ~0.35 to ~0.55 on SROIE.
"""
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

_TOTAL_KW_RE = re.compile(
    r"\b(total|amount|grand|due|payable)\b", re.IGNORECASE,
)
_SUBTOTAL_KW_RE = re.compile(r"\bsub[\s\-]?total\b|\bsubtotal\b", re.IGNORECASE)


def _is_short_junk(text: str) -> bool:
    """True if ``text`` is too short/digit-heavy/punctuation-only to be a
    plausible company name or address line."""
    s = text.strip()
    if len(s) < 3:
        return True
    alpha = sum(c.isalpha() for c in s)
    return alpha < 3


def _money_num(s: str) -> float:
    """Parse a money substring like ``'1,234.56'`` → ``1234.56``; 0 on failure."""
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return 0.0


def extract_total(
    region_texts: list[str], bbox_list: list[list[float]],
) -> tuple[int, str] | None:
    """Pick best TOTAL region: last money line whose neighbourhood has a total
    keyword (excluding subtotal). Falls back to keyword-ranked scoring."""
    # Build candidate list of (index, money_value) for all money-bearing lines.
    money_candidates: list[tuple[int, str]] = []
    for i, txt in enumerate(region_texts):
        m = _MONEY_RE.search(txt)
        if not m:
            continue
        money_candidates.append((i, m.group(0).strip()))

    if not money_candidates:
        return None

    # Primary heuristic: last money line whose 2-line neighbourhood (±1 line)
    # contains a total keyword but not a subtotal keyword.
    for i, raw_val in reversed(money_candidates):
        neighbourhood = " ".join(
            region_texts[max(0, i - 1): i + 2],
        )
        if _TOTAL_KW_RE.search(neighbourhood) and not _SUBTOTAL_KW_RE.search(neighbourhood):
            value = re.sub(r"^(RM|USD|SGD|MYR|\$)\s*", "", raw_val, flags=re.IGNORECASE)
            return i, value

    # Fallback: keyword-ranked scoring (original heuristic).
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
    top_score = max(c[0] for c in candidates)
    top = [c for c in candidates if c[0] == top_score]
    if top_score >= 3:
        best = max(top, key=lambda c: c[2])
    elif top_score == 0:
        best = max(candidates, key=lambda c: c[2])
    else:
        freq: dict[str, int] = {}
        for _, _, _, v in top:
            freq[v] = freq.get(v, 0) + 1
        best = max(top, key=lambda c: (freq[c[3]], _money_num(c[3]), c[2]))
    value = re.sub(r"^(RM|USD|SGD|MYR|\$)\s*", "", best[3], flags=re.IGNORECASE)
    return best[1], value


def extract_date(region_texts: list[str]) -> tuple[int, str] | None:
    """First region matching DATE_RE → (index, date-substring)."""
    for i, txt in enumerate(region_texts):
        m = _DATE_RE.search(txt)
        if m:
            return i, m.group(0)
    return None


def _pick_company(
    region_texts: list[str], bbox_list: list[list[float]], used: set[int],
) -> tuple[int, str] | None:
    """Topmost region that is not junk, header, date, or money."""
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


def _pick_address(
    region_texts: list[str], bbox_list: list[list[float]],
    used: set[int], company_y: float, total_y: float, date_y: float,
    max_lines: int = 6,
) -> str:
    """Concatenate address lines between company and first money/date region."""
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
        if _is_short_junk(t) or _DATE_RE.search(t) or _MONEY_RE.search(t):
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
        if _ADDR_EXCLUDE.search(t) or _HEADER_JUNK.match(t) or _is_short_junk(t):
            continue
        fallback.append(t)
        if len(fallback) >= 4:
            break
    return " ".join(fallback)
