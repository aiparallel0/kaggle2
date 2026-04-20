"""Spatial + regex baseline for receipt KIE — no ML dependencies.

Strategy (all field-specific):

* **date**  — first region whose text matches DATE_RE.
* **total** — keyword-aware ranking; prefers ``GRAND TOTAL`` / ``AMOUNT DUE``
  over bare ``TOTAL`` over any other money figure, and penalises lines
  containing ``CHANGE`` / ``SUBTOTAL`` / ``ROUNDING`` (common false positives).
* **company** — top-most non-junk non-header region.
* **address** — lines spatially between the company and the first money /
  date region, excluding phone / tax-ID lines.

These priors collectively lift a pure-spatial baseline from F1 ≈ 0.35 to
≈ 0.55 on the SROIE test split.
"""
from __future__ import annotations

from models.rule_fields import _pick_address, _pick_company, extract_date, extract_total
from models.rule_regex import DATE_RE, MONEY_RE

__all__ = ["DATE_RE", "MONEY_RE", "rule_based_assign"]


def rule_based_assign(
    region_texts: list[str], bbox_list: list[list[float]],
) -> dict[str, str]:
    """Assign company/date/address/total from text + normalised bboxes.

    Returns a dict keyed by field name with the matched substring value.
    May be empty or missing keys when a receipt has no usable regions —
    callers should handle that gracefully.
    """
    assigned: dict[str, str] = {}
    used: set[int] = set()

    date_pick = extract_date(region_texts)
    if date_pick is not None:
        i, v = date_pick
        assigned["date"] = v
        used.add(i)

    total_pick = extract_total(region_texts, bbox_list)
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

    total_y = (
        bbox_list[total_pick[0]][1]
        if total_pick is not None and total_pick[0] < len(bbox_list) else 0.0
    )
    date_y = (
        bbox_list[date_pick[0]][1]
        if date_pick is not None and date_pick[0] < len(bbox_list) else 0.0
    )
    addr = _pick_address(region_texts, bbox_list, used, company_y, total_y, date_y)
    if addr:
        assigned["address"] = addr
    return assigned
