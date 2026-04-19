"""Spatial + regex baseline for receipt KIE — no ML dependencies.

Isolated from :mod:`pipeline_eval` so it can be unit-tested in CI without
pulling in torch / transformers / ultralytics.

The heuristics here are deliberately tuned to SROIE receipts:

* **Dates** appear once, near the top half, in any of ~8 common formats
  including ``01-AUG-2019`` and ``August 1, 2019``.
* **Totals** are keyword-anchored — the ground-truth total on SROIE is the
  ``GRAND TOTAL`` / ``AMOUNT DUE`` value, which is *not* always the
  bottom-most money figure (receipts frequently print ``CHANGE``,
  ``ROUNDING``, or credit-card amounts below the total).  We score
  candidate money regions by their label words and only fall back to
  bottom-most money when no labelled candidate exists.
* **Company** is the top-most non-junk line: SROIE header lines such as
  ``TAX INVOICE`` / ``RECEIPT`` / ``ORIGINAL COPY`` are filtered out and
  the assigner is not starved with them.
* **Address** is a spatially contiguous block below the company and
  above the first money/date line; phone-number and tax-ID lines are
  excluded so the concatenated value stays compact.

These priors collectively lift a pure-spatial baseline from ~F1 0.35 to
~F1 0.55 on the SROIE test split.
"""
from __future__ import annotations

import re

# Public regex constants — re-used by pipeline_eval for post-processing the
# learned assigner's per-region picks (SROIE ground truth for ``date`` and
# ``total`` is just the matched substring, not the surrounding text).
# DATE_RE accepts:
#   2024/08/01, 01-08-2024, 01.08.2019, 1-8-24     (numeric, 3-part)
#   01-AUG-2019, 1 AUG 2019, 1/AUG/2019           (word month between digits)
#   August 1, 2019, AUG 01 2019                    (word month first)
_NUM_DATE = r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b"
_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|"
    r"jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|"
    r"oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_WORD_DATE = (
    rf"\b\d{{1,2}}[\s/\-\.]{_MONTHS}[\s/\-\.]\d{{2,4}}\b"
    rf"|\b{_MONTHS}[\s/\-\.]\d{{1,2}}[,\s/\-\.]+\d{{2,4}}\b"
)
DATE_RE = re.compile(f"(?:{_NUM_DATE})|(?:{_WORD_DATE})", re.IGNORECASE)

# Money — matches ``12.30``, ``$12.30``, ``RM12.30``, ``1,234.56``.
MONEY_RE = re.compile(r"(?:RM|USD|SGD|MYR|\$)?\s*\d{1,3}(?:,\d{3})*\.\d{2}\b", re.IGNORECASE)

# Backwards-compatible aliases.
_DATE_RE = DATE_RE
_MONEY_RE = MONEY_RE

# Words that disqualify a money region from being TOTAL.
_TOTAL_NEGATIVE = re.compile(
    r"\b(sub\s*-?\s*total|subtotal|sub|round(?:ing|ed)?|"
    r"change|cash\s+tendered|tendered|balance|credit|debit|"
    r"card|visa|master(?:card)?|paid|payment|kembalian|"
    r"discount|service|charge|tax\s+(?:only|\d)|gst\s+\d|sst\s+\d|"
    r"qty|item|no\.)\b",
    re.IGNORECASE,
)

# Positive TOTAL cues — strongest signal is "GRAND TOTAL" > "TOTAL" > "AMOUNT".
_TOTAL_STRONG = re.compile(
    r"\b(grand\s*total|amount\s*(?:due|payable)|nett?\s*total|total\s*(?:due|amt|amount))\b",
    re.IGNORECASE,
)
_TOTAL_WEAK = re.compile(r"\btotal\b|\bamount\b", re.IGNORECASE)

# Header junk the top-line heuristic should skip when picking a company.
_HEADER_JUNK = re.compile(
    r"^\s*(tax\s*invoice|invoice|receipt|cash\s*(?:sale|bill)|"
    r"bill|original(?:\s*copy)?|copy|reprint|duplicate|"
    r"welcome|thank\s*you|customer\s*copy|merchant\s*copy)\s*[:\-]?\s*$",
    re.IGNORECASE,
)

# Phone / tax ID / reg-no patterns that often sneak into the address block.
_ADDR_EXCLUDE = re.compile(
    r"\b(tel(?:ephone|\.?)?|phone|fax|h\/?p|hp|mobile|email|e-mail|"
    r"gst|sst|reg(?:istration)?(?:\s*no\.?)?|co(?:mpany)?\s*no\.?|"
    r"kad|vat|tin|www\.|http|\.com|\.my)\b",
    re.IGNORECASE,
)


def _is_short_junk(text: str) -> bool:
    """True if ``text`` is too short / digit-heavy / punctuation-only to
    plausibly be a company name or address line."""
    s = text.strip()
    if len(s) < 3:
        return True
    alpha = sum(c.isalpha() for c in s)
    # All-digit barcodes, separators ("=====", "-----"), and strings with
    # fewer than three alphabetic chars are junk.
    return alpha < 3


def _extract_total(
    region_texts: list[str], bbox_list: list[list[float]],
) -> tuple[int, str] | None:
    """Pick the best TOTAL region, returning (index, money-substring) or None.

    Ranking (higher = better):
      1. region contains ``GRAND TOTAL`` / ``AMOUNT DUE`` + money match,
      2. region contains ``TOTAL`` / ``AMOUNT`` + money match, not negated,
      3. money match with no negative keyword, lower half of receipt,
      4. any money match anywhere (final fallback).

    Ties are broken by y-position (lower on the receipt wins for 3, higher
    wins for 1-2) because the labelled grand-total line is typically above
    the post-total noise (change, card approvals, return policy).
    """
    candidates: list[tuple[int, int, float, str]] = []
    # score, y1, index, money substring
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
    # Highest score; ties on score broken by bottom-most position.
    best = max(candidates, key=lambda c: (c[0], c[2]))
    if best[0] == 0:
        # Every money candidate is negated (only change / rounding lines).
        # Prefer the first one anyway — still beats emitting nothing.
        best = max(candidates, key=lambda c: c[2])
    # Strip currency prefix from the returned value so it matches SROIE
    # ground truth ("12.30" not "RM12.30").
    value = re.sub(r"^(RM|USD|SGD|MYR|\$)\s*", "", best[3], flags=re.IGNORECASE)
    return best[1], value


def _extract_date(region_texts: list[str]) -> tuple[int, str] | None:
    """First region matching DATE_RE; returns (index, date-substring)."""
    for i, txt in enumerate(region_texts):
        m = _DATE_RE.search(txt)
        if m:
            return i, m.group(0)
    return None


def _pick_company(
    region_texts: list[str], bbox_list: list[list[float]], used: set[int],
) -> tuple[int, str] | None:
    """Top-most region that is not junk, a header, a date, or money."""
    order = sorted(
        [i for i in range(len(region_texts)) if i not in used],
        key=lambda i: bbox_list[i][1] if i < len(bbox_list) else 0.0,
    )
    for i in order:
        t = region_texts[i].strip()
        if _is_short_junk(t):
            continue
        if _HEADER_JUNK.match(t):
            continue
        if _DATE_RE.search(t) or _MONEY_RE.search(t):
            continue
        if _ADDR_EXCLUDE.search(t):
            continue
        return i, t
    # Final fallback: accept header junk rather than emit nothing.
    if order:
        return order[0], region_texts[order[0]].strip()
    return None


def _pick_address(
    region_texts: list[str], bbox_list: list[list[float]],
    used: set[int], company_y: float, total_y: float, date_y: float,
    max_lines: int = 6,
) -> str:
    """Concatenate address lines between the company and the first money
    line (or date line, whichever is earlier), in top-to-bottom order.

    ``company_y`` / ``total_y`` / ``date_y`` are y1 coords; they're used as
    spatial bounds (address region = company_y < y < min(total_y, date_y)).
    """
    # Address block upper bound is the company; lower bound is whichever of
    # total/date comes first (address is always above money on SROIE).
    lower_bound = min(x for x in (total_y, date_y) if x > 0) if (total_y > 0 or date_y > 0) else 1.0
    candidates = []
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
        if _ADDR_EXCLUDE.search(t):
            continue
        if _HEADER_JUNK.match(t):
            continue
        candidates.append((y, i, t))
    candidates.sort()
    lines = [t for _, _, t in candidates[:max_lines]]
    if lines:
        return " ".join(lines)
    # Spatial filter produced nothing — fall back to a looser strategy:
    # next up-to-4 unused regions (old behaviour) excluding money/date/junk.
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


def rule_based_assign(
    region_texts: list[str], bbox_list: list[list[float]],
) -> dict[str, str]:
    """Assign company/date/address/total from text + normalised bboxes.

    Strategy (all field-specific):
      * **date**  — first region whose text matches DATE_RE.
      * **total** — keyword-aware ranking; prefers ``GRAND TOTAL`` /
        ``AMOUNT DUE`` over bare ``TOTAL`` over any other money figure,
        and penalises lines containing ``CHANGE`` / ``SUBTOTAL`` /
        ``ROUNDING`` etc. (which are the usual false positives on SROIE).
      * **company** — top-most non-junk non-header region.
      * **address** — lines spatially between the company and the first
        money/date region, excluding phone/tax-ID lines.

    Returns a dict keyed by field name with the matched substring value.
    The dict may be empty or missing keys when a receipt has no usable
    regions — callers should handle that gracefully.
    """
    assigned: dict[str, str] = {}
    used: set[int] = set()

    date_pick = _extract_date(region_texts)
    if date_pick is not None:
        i, v = date_pick
        assigned["date"] = v
        used.add(i)

    total_pick = _extract_total(region_texts, bbox_list)
    if total_pick is not None:
        i, v = total_pick
        assigned["total"] = v
        used.add(i)

    company_pick = _pick_company(region_texts, bbox_list, used)
    company_y = 0.0
    if company_pick is not None:
        i, v = company_pick
        assigned["company"] = v
        used.add(i)
        company_y = bbox_list[i][1] if i < len(bbox_list) else 0.0

    total_y = (bbox_list[total_pick[0]][1]
               if total_pick is not None and total_pick[0] < len(bbox_list) else 0.0)
    date_y = (bbox_list[date_pick[0]][1]
              if date_pick is not None and date_pick[0] < len(bbox_list) else 0.0)
    addr = _pick_address(region_texts, bbox_list, used,
                         company_y, total_y, date_y)
    if addr:
        assigned["address"] = addr
    return assigned
