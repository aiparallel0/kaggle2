"""Extract labeled crops and per-receipt region groups from SROIE box files."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.types import Crop, Receipt


def _match_field(text: str, gt: dict[str, str]) -> str:
    """Match a box text line to the best KIE field by overlap / substring."""
    low = text.lower().strip()
    if not low:
        return ""
    addr = gt.get("address", "").lower()
    if addr and low in addr and len(low) >= 3:
        return "address"
    tokens = set(low.split())
    best, best_score = "", 0
    for name, value in gt.items():
        overlap = len(tokens & set(value.lower().split()))
        if overlap > best_score:
            best, best_score = name.lower(), overlap
    return best


def _parse_box_file(rec: Receipt, fields: list[str]) -> list[Crop]:
    box_path = rec.image_path.parent.parent / "box" / (rec.image_path.stem + ".txt")
    if not box_path.exists():
        return []
    gt = {f.name.lower(): f.value for f in rec.fields}
    ok = {f.lower() for f in fields}
    with Image.open(rec.image_path) as img:
        w, h = img.size
    out: list[Crop] = []
    for line in box_path.read_text(errors="replace").splitlines():
        parts = line.split(",", 8)
        if len(parts) < 9:
            continue
        try:
            coords = [int(p) for p in parts[:8]]
        except ValueError:
            continue
        text = parts[8].strip()
        if not text:
            continue
        xs, ys = coords[0::2], coords[1::2]
        x1, y1 = max(0.0, min(xs) / w), max(0.0, min(ys) / h)
        x2, y2 = min(1.0, max(xs) / w), min(1.0, max(ys) / h)
        if x2 <= x1 or y2 <= y1:
            continue
        raw = _match_field(text, gt)
        label = raw if raw in ok else ""
        out.append(Crop(
            image_path=Path(rec.image_path), bbox=(x1, y1, x2, y2),
            text=text, field_label=label,
        ))
    return out


def extract_crops(receipts: list[Receipt], fields: list[str]) -> list[Crop]:
    """Parse SROIE box annotations → labeled Crop list for TrOCR + assigner."""
    return [c for r in receipts for c in _parse_box_file(r, fields) if c.field_label]


def extract_receipt_regions(
    receipts: list[Receipt], fields: list[str],
) -> list[list[Crop]]:
    """Parse annotations → ALL box regions per receipt (labeled + distractors)."""
    gs = [_parse_box_file(r, fields) for r in receipts]
    return [g for g in gs if any(c.field_label for c in g)]
