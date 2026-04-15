"""Download SROIE dataset, produce train/val/test splits, extract crops."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from core.errors import DataError
from core.types import Crop, DataSplit, ExpConfig, Field, Receipt

# 500 train / 63 val / 63 test from 626 SROIE images (Bug 7: val ≠ test)
_N_VAL = 63
_N_TEST = 63


def download_sroie(config: ExpConfig) -> Path:
    """Clone SROIE repo if needed; return path to data directory."""
    cache = Path(config.data_dir)
    if (cache / "train" / "img").exists():
        return cache
    cache.mkdir(parents=True, exist_ok=True)
    tmp = cache / "_git"
    try:
        subprocess.run(
            ["git", "clone", "--depth=1", config.sroie_url, str(tmp)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as exc:
        raise DataError(f"SROIE clone failed: {exc.stderr.decode()}") from exc
    for split in ("train", "test"):
        src_img = tmp / "data" / split / "img"
        src_box = tmp / "data" / split / "box"
        src_ent = tmp / "data" / split / "entities"
        for src, name in [(src_img, "img"), (src_box, "box"), (src_ent, "entities")]:
            if src.exists():
                dst = cache / split / name
                dst.mkdir(parents=True, exist_ok=True)
                for f in src.iterdir():
                    shutil.copy(f, dst / f.name)
    shutil.rmtree(tmp, ignore_errors=True)
    return cache


def _load_receipts(img_dir: Path, ent_dir: Path) -> list[Receipt]:
    receipts: list[Receipt] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        ent_path = ent_dir / img_path.with_suffix(".json").name
        if not ent_path.exists():
            ent_path = ent_dir / (img_path.stem + ".txt")
        if not ent_path.exists():
            continue
        try:
            raw = json.loads(ent_path.read_text())
        except json.JSONDecodeError:
            raw = _parse_entities_txt(ent_path)
        fields = [Field(name=k, value=str(v)) for k, v in raw.items()]
        receipts.append(Receipt(image_path=img_path, fields=fields))
    return receipts


def _parse_entities_txt(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip().lower()] = v.strip()
    return result


def _match_field(text: str, gt: dict[str, str]) -> str:
    """Match a box text line to the best-matching KIE field by token overlap."""
    tokens = set(text.lower().split())
    best, best_score = "", 0
    for name, value in gt.items():
        vtokens = set(value.lower().split())
        overlap = len(tokens & vtokens)
        if overlap > best_score:
            best, best_score = name, overlap
    return best


def extract_crops(receipts: list[Receipt], fields: list[str]) -> list[Crop]:
    """Parse SROIE box annotations → labeled Crop list for TrOCR + assigner."""
    crops: list[Crop] = []
    for rec in receipts:
        box_path = rec.image_path.parent.parent / "box" / (rec.image_path.stem + ".txt")
        if not box_path.exists():
            continue
        gt = {f.name.lower(): f.value for f in rec.fields}
        img = Image.open(rec.image_path)
        w, h = img.size
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
            x1 = max(0, min(coords[0], coords[6])) / w
            y1 = max(0, min(coords[1], coords[3])) / h
            x2 = min(w, max(coords[2], coords[4])) / w
            y2 = min(h, max(coords[5], coords[7])) / h
            label = _match_field(text, gt)
            if label and label in fields:
                crops.append(Crop(
                    image_path=rec.image_path,
                    bbox=(x1, y1, x2, y2),
                    text=text,
                    field_label=label,
                ))
    return crops


def split_sroie(data_path: Path, seed: int) -> DataSplit:
    """Split SROIE into train/val/test (Bug 7: physically separate val/test)."""
    img_dir = data_path / "train" / "img"
    ent_dir = data_path / "train" / "entities"
    if not img_dir.exists():
        raise DataError(f"SROIE train/img not found at {img_dir}")
    all_receipts = _load_receipts(img_dir, ent_dir)
    rng = random.Random(seed)
    rng.shuffle(all_receipts)
    test = all_receipts[:_N_TEST]
    val = all_receipts[_N_TEST: _N_TEST + _N_VAL]
    train = all_receipts[_N_TEST + _N_VAL:]
    # Bug 7: assert zero overlap between val and test sets
    val_ids = {r.image_path.stem for r in val}
    test_ids = {r.image_path.stem for r in test}
    assert len(val_ids & test_ids) == 0, "Val/test overlap detected"
    return DataSplit(train=train, val=val, test=test)
