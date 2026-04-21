"""Spatial + regex rule-based baseline for receipt KIE (no ML deps).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: provides the rule-based assignment arm that serves as an ablation
    lower bound.  Strategies: date=first DATE_RE match, total=keyword-ranked
    money, company=topmost non-junk, address=spatially between company/total.
"""
from __future__ import annotations

from models.rule_fields import _pick_address, _pick_company, extract_date, extract_total
from models.rule_regex import DATE_RE, MONEY_RE

__all__ = ["DATE_RE", "MONEY_RE", "rule_based_assign"]


def rule_based_assign(
    region_texts: list[str], bbox_list: list[list[float]],
) -> dict[str, str]:
    """Assign company/date/address/total using spatial + regex heuristics."""
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
