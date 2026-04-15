"""Train YOLO on SROIE bounding-box layout for text-region detection."""
from __future__ import annotations

import os
import shutil
import textwrap
from pathlib import Path

from core.errors import TrainError
from core.types import DataSplit, ExpConfig, Receipt

# YOLO expects images at yolo_data/images/{train,val}/ and
# labels at yolo_data/labels/{train,val}/ in YOLO-format txt files.


def _write_yolo_labels(receipts: list[Receipt], img_dst: Path, lbl_dst: Path) -> None:
    """Copy images + write placeholder full-image labels (YOLO txt format)."""
    for r in receipts:
        shutil.copy(r.image_path, img_dst / r.image_path.name)
        # Full-image label: class 0, centre 0.5 0.5, w 1.0, h 1.0
        lbl = lbl_dst / (r.image_path.stem + ".txt")
        lbl.write_text("0 0.5 0.5 1.0 1.0\n")


def train_yolo(config: ExpConfig, data: DataSplit) -> str:
    """Train YOLOv8 on SROIE; return path to best.pt weights."""
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise TrainError("ultralytics not installed — pip install ultralytics") from exc

    base = Path(config.output_dir) / "yolo_data"
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)

    _write_yolo_labels(data.train, base / "images" / "train", base / "labels" / "train")
    _write_yolo_labels(data.val, base / "images" / "val", base / "labels" / "val")

    yaml_path = base / "dataset.yaml"
    yaml_path.write_text(
        textwrap.dedent(f"""\
        path: {base.resolve()}
        train: images/train
        val: images/val
        nc: 1
        names: [receipt]
        """)
    )

    out_dir = os.path.join(config.output_dir, "yolo")
    model = YOLO(config.yolo_model)
    model.train(
        data=str(yaml_path),
        epochs=config.epochs_yolo,
        imgsz=config.yolo_img_size,  # Bug 5: always pass imgsz explicitly
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
