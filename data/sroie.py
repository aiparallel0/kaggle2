"""Download SROIE dataset, produce train/val/test splits.

Re-exports ``extract_crops`` / ``extract_receipt_regions`` / ``_match_field``
from ``sroie_crops`` so existing callers keep working.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

from core.errors import DataError
from core.types import DataSplit, ExpConfig, Field, Receipt
from data.sroie_crops import (
    _match_field,
    _parse_box_file,
    extract_crops,
    extract_receipt_regions,
)

__all__ = [
    "_match_field",
    "_parse_box_file",
    "download_sroie",
    "extract_crops",
    "extract_receipt_regions",
    "load_or_create_split",
    "split_sroie",
]

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

    # Hierarchical layout: data/{train,test}/{img,box,entities}.
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

    # Flat layout (zzzDavid/ICDAR-2019-SROIE): data/{img,box,key}. All files
    # go under train/; split_sroie() partitions them below.
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


def _parse_entities_txt(path: Path) -> dict[str, str]:
    """Parse key:value entity files, preserving colons that appear in values."""
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        # partition splits at the FIRST colon only; anything after (including
        # further colons like "12:30:00") is preserved in the value.
        k, _, v = line.partition(":")
        k, v = k.strip().lower(), v.strip()
        if k:
            out[k] = v
    return out


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


def split_sroie(data_path: Path, seed: int) -> DataSplit:
    """Split SROIE into train/val/test (Bug 7: physically separate val/test)."""
    img_dir, ent_dir = data_path / "train" / "img", data_path / "train" / "entities"
    if not img_dir.exists():
        raise DataError(f"SROIE train/img not found at {img_dir}")
    all_r = _load_receipts(img_dir, ent_dir)
    random.Random(seed).shuffle(all_r)
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
