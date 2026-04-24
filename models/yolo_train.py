"""Train YOLOv8n on SROIE bounding-box annotations for text-line detection.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: converts SROIE 4-corner-point box annotations to YOLO format and
    trains a single-class text-line detector.  Bug 5 and Bug 8 guardrails
    ensure imgsz consistency and absolute project path.
"""
from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

from PIL import Image

from core.errors import TrainError
from core.types import DataSplit, ExpConfig, Receipt

# YOLO expects images at yolo_data/images/{train,val}/ and
# labels at yolo_data/labels/{train,val}/ in YOLO-format txt files.


def _yolo_lines_from_sroie_box(box_path: Path, img_w: int, img_h: int) -> list[str]:
    """Convert SROIE 4-corner-point boxes → YOLO (class cx cy w h) format."""
    lines: list[str] = []
    for raw in box_path.read_text(errors="replace").splitlines():
        parts = raw.split(",", 8)
        if len(parts) < 8:
            continue
        try:
            coords = [int(p) for p in parts[:8]]
        except ValueError:
            continue
        xs, ys = coords[0::2], coords[1::2]
        xmin, xmax = max(0, min(xs)), min(img_w, max(xs))
        ymin, ymax = max(0, min(ys)), min(img_h, max(ys))
        if xmax <= xmin or ymax <= ymin:
            continue
        cx = (xmin + xmax) / 2.0 / img_w
        cy = (ymin + ymax) / 2.0 / img_h
        bw = (xmax - xmin) / img_w
        bh = (ymax - ymin) / img_h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    return lines


def _write_yolo_labels(receipts: list[Receipt], img_dst: Path, lbl_dst: Path) -> int:
    """Copy images; derive per-text-line YOLO labels from SROIE box/ annotations.

    Returns number of images that produced at least one real label.
    """
    labelled = 0
    for r in receipts:
        shutil.copy(r.image_path, img_dst / r.image_path.name)
        lbl = lbl_dst / (r.image_path.stem + ".txt")
        box_path = r.image_path.parent.parent / "box" / (r.image_path.stem + ".txt")
        if not box_path.exists():
            # Missing annotations: write an empty file so YOLO treats it as background.
            lbl.write_text("")
            continue
        with Image.open(r.image_path) as img:
            w, h = img.size
        lines = _yolo_lines_from_sroie_box(box_path, w, h)
        lbl.write_text("\n".join(lines) + ("\n" if lines else ""))
        if lines:
            labelled += 1
    return labelled


def train_yolo(config: ExpConfig, data: DataSplit) -> str:
    """Train YOLOv8n on SROIE with Bug 5/8 guards; return best.pt path."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainError("ultralytics not installed — pip install ultralytics") from exc

    base = Path(config.output_dir) / "yolo_data"
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = _write_yolo_labels(
        data.train, base / "images" / "train", base / "labels" / "train"
    )
    n_val = _write_yolo_labels(data.val, base / "images" / "val", base / "labels" / "val")
    if n_train == 0:
        raise TrainError(
            "YOLO: zero training images have SROIE box annotations — "
            "cannot train a text-region detector from placeholder labels."
        )
    print(f"YOLO: labelled {n_train}/{len(data.train)} train, {n_val}/{len(data.val)} val")

    yaml_path = base / "dataset.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
        path: {base.resolve()}
        train: images/train
        val: images/val
        nc: 1
        names: [text_line]
        """)
    )

    # Bug 8: ultralytics >=8.3 resolves a relative ``project=`` against its
    # internal settings ``runs_dir`` (defaults to ``runs/detect/``), not
    # against CWD.  Passing ``project="./results/yolo"`` therefore writes to
    # ``./runs/detect/results/yolo/run/weights/best.pt`` — which our caller
    # never finds.  Resolve to an absolute path before handing it over so
    # the project root and the lookup path agree byte-for-byte.
    # Bug 8 (gate): resolve project= to an absolute path so ultralytics
    # doesn't write under runs/detect/… and then fail to find best.pt.
    # Guard off = pass the raw (relative) path through to reproduce the
    # original FileExistsError / stale-checkpoint failure mode.
    if config.bug_flags.get("bug_8", True):
        out_dir = str(Path(config.output_dir, "yolo").resolve())
    else:
        out_dir = str(Path(config.output_dir, "yolo"))
    model = YOLO(config.yolo_model)
    # Bug 5 (gate): explicit imgsz= at train.  Off = use ultralytics default.
    _imgsz = config.yolo_img_size if config.bug_flags.get("bug_5", True) else 640
    model.train(
        data=str(yaml_path),
        epochs=config.epochs_yolo,
        imgsz=_imgsz,
        batch=config.batch_size,
        seed=config.seed,
        project=out_dir,
        name="run",
        exist_ok=True,
    )
    best = os.path.join(out_dir, "run", "weights", "best.pt")
    if not Path(best).exists():
        raise TrainError(f"YOLO training finished but best.pt not found at {best}")
    return best
