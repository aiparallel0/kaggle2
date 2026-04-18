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

    # Try the hierarchical layout first (data/{train,test}/{img,box,entities}).
    found_hierarchical = False
    for split in ("train", "test"):
        for name in ("img", "box", "entities"):
            src = tmp / "data" / split / name
            if not src.exists():
                continue
            found_hierarchical = True
            dst = cache / split / name
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                shutil.copy(f, dst / f.name)

    # Fall back to the flat layout (data/{img,box,key}) used by the
    # zzzDavid/ICDAR-2019-SROIE repo.  All files go under train/ because
    # split_sroie() partitions them into train/val/test later.
    if not found_hierarchical:
        _FLAT_MAP: dict[str, str] = {"img": "img", "box": "box", "key": "entities"}
        for src_name, dst_name in _FLAT_MAP.items():
            src = tmp / "data" / src_name
            if not src.exists():
                continue
            dst = cache / "train" / dst_name
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                # _parse_box_file expects .txt; the repo ships .csv
                out_name = (
                    f.with_suffix(".txt").name
                    if dst_name == "box" and f.suffix == ".csv"
                    else f.name
                )
                shutil.copy(f, dst / out_name)

    shutil.rmtree(tmp, ignore_errors=True)
    return cache


def _load_receipts(img_dir: Path, ent_dir: Path) -> list[Receipt]:
    receipts: list[Receipt] = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        ent = ent_dir / img_path.with_suffix(".json").name
        if not ent.exists():
            ent = ent_dir / (img_path.stem + ".txt")
        if not ent.exists():
            continue
        try:
            raw = json.loads(ent.read_text())
        except json.JSONDecodeError:
            raw = _parse_entities_txt(ent)
        receipts.append(Receipt(
            image_path=img_path,
            fields=[Field(name=k, value=str(v)) for k, v in raw.items()],
        ))
    return receipts


def _parse_entities_txt(path: Path) -> dict[str, str]:
    """Parse key:value entity files, preserving colons that appear in values."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        # partition splits at the FIRST colon only; everything after (including
        # any further colons) is preserved in the value — e.g. "12:30:00".
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k:  # skip lines that start with ':' (empty key after strip)
            out[k] = v
    return out


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
        out.append(Crop(image_path=rec.image_path, bbox=(x1, y1, x2, y2),
                        text=text, field_label=label))
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


def split_sroie(data_path: Path, seed: int) -> DataSplit:
    """Split SROIE into train/val/test (Bug 7: physically separate val/test)."""
    img_dir, ent_dir = data_path / "train" / "img", data_path / "train" / "entities"
    if not img_dir.exists():
        raise DataError(f"SROIE train/img not found at {img_dir}")
    all_r = _load_receipts(img_dir, ent_dir)
    random.Random(seed).shuffle(all_r)
    # Conventional order: val slice first, then test, then train.
    val, test = all_r[:_N_VAL], all_r[_N_VAL: _N_VAL + _N_TEST]
    train = all_r[_N_VAL + _N_TEST:]
    assert not ({r.image_path.stem for r in val}
                & {r.image_path.stem for r in test}), "Val/test overlap"  # Bug 7
    return DataSplit(train=train, val=val, test=test)


def load_or_create_split(data_path: Path, seed: int, cache: Path) -> DataSplit:
    """Reuse the saved split if present, else create+persist (prevents drift)."""
    groups = ("train", "val", "test")
    if cache.exists():
        raw = json.loads(cache.read_text())
        by_stem = {r.image_path.stem: r for r in _load_receipts(
            data_path / "train" / "img", data_path / "train" / "entities")}
        miss = [s for g in groups for s in raw[g] if s not in by_stem]
        if miss:
            raise DataError(f"Saved split missing images: {miss[:5]}...")
        return DataSplit(*([by_stem[s] for s in raw[g]] for g in groups))
    split = split_sroie(data_path, seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {g: [r.image_path.stem for r in getattr(split, g)] for g in groups},
        indent=2,
    ))
    return split
