"""PR-E — Synthetic receipt generator for tiny/small Pareto-sweep cells.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: generates ``(bbox, text, label)`` triples that look like SROIE
    in distribution (4-field schema, multi-line address, money + date
    regex) so the tiny / small assigner sweep cells have enough
    training data to converge on a sub-receipt-count of 50.  No
    images — assigner training only consumes bboxes + text + label.

Determinism: every receipt is parameterised by an integer ``seed``
so the 50,000-receipt corpus is reproducible bit-for-bit across
machines.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

from core.types import ExpConfig

_FIELDS = ("company", "address", "date", "total")
_COMPANIES = (
    "RESTORAN A", "MART B", "CAFE C", "GROCER D", "PHARMACY E",
)
_STREETS = (
    "JALAN ALPHA", "LORONG BETA", "JALAN GAMMA", "LORONG DELTA",
)


@dataclass(frozen=True)
class SyntheticReceipt:
    """One synthetic receipt in (bbox, text, label) form."""

    receipt_id: str
    boxes: list[tuple[float, float, float, float]]
    texts: list[str]
    labels: list[str]


def generate_synthetic_receipts(
    n: int, config: ExpConfig,
) -> list[SyntheticReceipt]:
    """Generate ``n`` synthetic SROIE-shaped receipts.

    Each receipt has 6–14 lines; one ``company`` line at top, 2–4
    ``address`` lines, one ``date`` line, one ``total`` line, and the
    remainder are distractors (subtotal, cash, change, item lines).
    Bounding boxes are generated in normalised ``[0, 1]`` coordinates
    on a per-line height of ``1/n_lines``.
    """
    rng = random.Random(config.seed)
    out: list[SyntheticReceipt] = []
    for i in range(n):
        out.append(_one_receipt(f"synth_{i:06d}", rng))
    return out


def _one_receipt(rid: str, rng: random.Random) -> SyntheticReceipt:
    n_lines = rng.randint(6, 14)
    n_addr = rng.randint(2, 4)
    boxes: list[tuple[float, float, float, float]] = []
    texts: list[str] = []
    labels: list[str] = []
    h = 1.0 / n_lines
    for i in range(n_lines):
        y = i * h
        boxes.append((0.05, y, 0.95, y + h * 0.9))
    company = rng.choice(_COMPANIES)
    addr_street = rng.choice(_STREETS)
    date = (
        f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/"
        f"{rng.randint(2014, 2019)}"
    )
    total = f"{rng.randint(5, 999)}.{rng.randint(0, 99):02d}"
    texts.append(company)
    labels.append("company")
    for j in range(n_addr):
        texts.append(f"{addr_street} {j + 1}")
        labels.append("address")
    while len(texts) < n_lines - 2:
        texts.append("ITEM " + str(rng.randint(1, 99)))
        labels.append("distractor")
    texts.append(date)
    labels.append("date")
    texts.append(f"TOTAL RM {total}")
    labels.append("total")
    while len(labels) < len(boxes):
        labels.append("distractor")
    return SyntheticReceipt(rid, boxes[:len(texts)], texts, labels[:len(texts)])
