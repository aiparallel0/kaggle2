"""Download SROIE dataset and produce train/val/test splits."""
from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path

from core.errors import DataError
from core.types import DataSplit, ExpConfig, Field, Receipt

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
    return DataSplit(train=train, val=val, test=test)
