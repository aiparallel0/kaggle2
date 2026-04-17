"""Unit tests for SROIE parsing and crop extraction helpers."""
from __future__ import annotations

from pathlib import Path

from PIL import Image

from core.types import Field, Receipt
from data.sroie import _match_field, extract_crops, extract_receipt_regions


def _mk_receipt(tmp_path: Path, box_text: str, fields: dict[str, str]) -> Receipt:
    img_dir = tmp_path / "train" / "img"
    box_dir = tmp_path / "train" / "box"
    img_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "r1.jpg"
    Image.new("RGB", (1000, 500), "white").save(img_path)
    (box_dir / "r1.txt").write_text(box_text)
    return Receipt(
        image_path=img_path,
        fields=[Field(name=k, value=v) for k, v in fields.items()],
    )


def test_match_field_address_substring() -> None:
    gt = {"address": "123 main street springfield", "company": "acme"}
    assert _match_field("123 main street", gt) == "address"
    assert _match_field("main street springfield", gt) == "address"


def test_match_field_token_overlap_for_total() -> None:
    gt = {"total": "10.00", "company": "acme"}
    assert _match_field("TOTAL 10.00", gt) == "total"


def test_match_field_empty_returns_empty() -> None:
    assert _match_field("", {"company": "acme"}) == ""


def test_match_field_no_overlap_returns_empty() -> None:
    assert _match_field("unrelated gibberish", {"total": "10.00"}) == ""


def test_extract_crops_filters_unlabeled(tmp_path: Path) -> None:
    box = (
        "10,20,110,20,110,70,10,70,ACME\n"
        "10,80,110,80,110,130,10,130,UNRELATED TEXT\n"
        "10,400,110,400,110,450,10,450,10.00\n"
    )
    rec = _mk_receipt(tmp_path, box, {"company": "ACME", "total": "10.00"})
    crops = extract_crops([rec], ["company", "date", "address", "total"])
    labels = {c.field_label for c in crops}
    assert "company" in labels
    assert "total" in labels
    assert "" not in labels  # unlabeled crops are excluded


def test_extract_receipt_regions_keeps_distractors(tmp_path: Path) -> None:
    box = (
        "10,20,110,20,110,70,10,70,ACME\n"
        "10,80,110,80,110,130,10,130,UNRELATED\n"
    )
    rec = _mk_receipt(tmp_path, box, {"company": "ACME"})
    groups = extract_receipt_regions([rec], ["company", "date", "address", "total"])
    assert len(groups) == 1
    regions = groups[0]
    assert len(regions) == 2
    assert any(r.field_label == "company" for r in regions)
    assert any(r.field_label == "" for r in regions)


def test_extract_receipt_regions_bboxes_in_unit_range(tmp_path: Path) -> None:
    box = "10,20,110,20,110,70,10,70,ACME\n"
    rec = _mk_receipt(tmp_path, box, {"company": "ACME"})
    groups = extract_receipt_regions([rec], ["company"])
    x1, y1, x2, y2 = groups[0][0].bbox
    assert 0.0 <= x1 < x2 <= 1.0
    assert 0.0 <= y1 < y2 <= 1.0
