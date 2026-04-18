"""Spatial + regex baseline for receipt KIE — no ML dependencies.

Isolated from :mod:`pipeline_eval` so it can be unit-tested in CI without
pulling in torch / transformers / ultralytics.
"""
from __future__ import annotations

import re

# Public regex constants — re-used by pipeline_eval for post-processing
# the learned assigner's per-region picks (ground truth for SROIE ``date``
# and ``total`` is just the matched substring, not the surrounding text).
DATE_RE = re.compile(r"\b\d{1,4}[/\-\.]\d{1,2}[/\-\.]\d{1,4}\b")
MONEY_RE = re.compile(r"\$?\d+(?:,\d{3})*\.\d{2}\b")
# Backwards-compatible aliases (used internally below).
_DATE_RE = DATE_RE
_MONEY_RE = MONEY_RE


def rule_based_assign(
    region_texts: list[str], bbox_list: list[list[float]],
) -> dict[str, str]:
    """Assign company/date/address/total from text + normalised bboxes.

    Strategy:
      * date: first region matching a date regex,
      * total: bottom-most (largest y1) region matching a money regex,
      * company: top-most (smallest y1) unused region,
      * address: next up-to-4 unused regions, skipping money/date lines.

    Returns a dict keyed by field name with the matched substring value.
    """
    assigned: dict[str, str] = {}
    used: set[int] = set()
    for i, txt in enumerate(region_texts):
        m = _DATE_RE.search(txt)
        if m:
            assigned["date"] = m.group(0)
            used.add(i)
            break
    money = [
        (i, bbox_list[i][3], _MONEY_RE.search(region_texts[i]))
        for i in range(len(region_texts))
        if i not in used and _MONEY_RE.search(region_texts[i].strip())
    ]
    if money:
        best = max(money, key=lambda x: x[1])
        assigned["total"] = best[2].group(0)  # type: ignore[union-attr]
        used.add(best[0])
    by_y = sorted(
        [(i, bbox_list[i][1]) for i in range(len(region_texts)) if i not in used],
        key=lambda x: x[1],
    )
    if by_y:
        assigned["company"] = region_texts[by_y[0][0]]
        used.add(by_y[0][0])
    addr: list[str] = []
    for i, _ in by_y[1:]:
        t = region_texts[i].strip()
        if _MONEY_RE.search(t) or _DATE_RE.search(t):
            continue
        addr.append(t)
        if len(addr) >= 4:
            break
    if addr:
        assigned["address"] = " ".join(addr)
    return assigned
