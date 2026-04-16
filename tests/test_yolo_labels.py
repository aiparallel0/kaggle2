"""Unit tests for SROIE → YOLO label conversion (realism fix for train_yolo)."""
from __future__ import annotations

from models.yolo_train import _yolo_lines_from_sroie_box


def test_four_corner_line_converts_to_normalised_box(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "box.txt"
    # SROIE: x1,y1,x2,y2,x3,y3,x4,y4,text  (axis-aligned 100x50 box at (10,20))
    p.write_text("10,20,110,20,110,70,10,70,HELLO\n")
    lines = _yolo_lines_from_sroie_box(p, img_w=1000, img_h=500)
    assert len(lines) == 1
    cls, cx, cy, bw, bh = lines[0].split()
    assert cls == "0"
    assert abs(float(cx) - 0.060) < 1e-6   # (10+110)/2 / 1000
    assert abs(float(cy) - 0.090) < 1e-6   # (20+70)/2 / 500
    assert abs(float(bw) - 0.100) < 1e-6   # 100/1000
    assert abs(float(bh) - 0.100) < 1e-6   # 50/500


def test_degenerate_and_malformed_lines_dropped(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "box.txt"
    p.write_text(
        "\n"
        "not,enough,fields\n"
        "a,b,c,d,e,f,g,h,text\n"            # non-integer coords → dropped
        "10,20,10,20,10,20,10,20,zero\n"    # zero-area → dropped
        "0,0,100,0,100,50,0,50,valid\n"
    )
    lines = _yolo_lines_from_sroie_box(p, img_w=1000, img_h=500)
    assert len(lines) == 1
    assert lines[0].startswith("0 ")


def test_out_of_bounds_coords_are_clipped(tmp_path):  # type: ignore[no-untyped-def]
    p = tmp_path / "box.txt"
    # Coordinates extend past image edge; conversion must clip to image bounds.
    p.write_text("-10,-10,2000,-10,2000,1000,-10,1000,OVERFLOW\n")
    lines = _yolo_lines_from_sroie_box(p, img_w=1000, img_h=500)
    assert len(lines) == 1
    _, cx, cy, bw, bh = lines[0].split()
    assert 0.0 <= float(cx) <= 1.0
    assert 0.0 <= float(cy) <= 1.0
    assert 0.0 < float(bw) <= 1.0
    assert 0.0 < float(bh) <= 1.0
