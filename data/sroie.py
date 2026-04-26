"""Download SROIE, produce the 500/63/63 train/val/test split (Bug 7 guard).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: clones the SROIE repository, parses JSON/TXT entity files, and
    partitions 626 receipts into disjoint val/test splits to prevent the
    silent F1-destroying Bug 7 (val≡test leakage).
"""
from __future__ import annotations

import json
import logging
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
    """Clone SROIE repo if absent; return data directory path."""
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
    """Parse key:value entity files; preserve colons inside values."""
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


def split_sroie(data_path: Path, seed: int, bug_flags: dict[str, bool] | None = None) -> DataSplit:
    """Partition SROIE into 500/63/63 train/val/test (Bug 7: disjoint sets).

    ``bug_flags`` is optional for back-compat; when ``bug_flags["bug_7"]``
    is False the val/test overlap assertion is skipped, reintroducing the
    leakage bug for ablation runs.  Defaults to all-on when None.
    """
    img_dir, ent_dir = data_path / "train" / "img", data_path / "train" / "entities"
    if not img_dir.exists():
        raise DataError(f"SROIE train/img not found at {img_dir}")
    all_r = _load_receipts(img_dir, ent_dir)
    random.Random(seed).shuffle(all_r)
    val, test = all_r[:_N_VAL], all_r[_N_VAL: _N_VAL + _N_TEST]
    train = all_r[_N_VAL + _N_TEST:]
    # Bug 7 (gate): val/test overlap assert.  With the guard active the
    # lists are already disjoint by construction; the assert is a belt-
    # and-braces catch.  Skipping it has no effect on THIS construction
    # but the gate remains in case a future refactor reintroduces drift.
    if bug_flags is None or bug_flags.get("bug_7", True):
        assert not ({r.image_path.stem for r in val}
                    & {r.image_path.stem for r in test}), "Val/test overlap"
    return DataSplit(train=train, val=val, test=test)


def _canonical_test_split(
    data_path: Path, config: ExpConfig,
) -> DataSplit | None:
    """Return canonical 347-image SROIE test split when test labels exist.

    When ``config.canonical_sroie_enabled`` is True this function will
    auto-download the ICDAR-2019 SROIE Task-3 test archive (347 images
    + GT) into ``data_path/test/{img,entities}/`` via
    :func:`data.sroie_canonical.ensure_canonical_test_set` (sha256-
    verified, with a docTR mirror fallback) before reading.  All 626
    training images become the train pool; the canonical 347 images
    become the test set; ``_N_VAL`` images are reserved from the
    training pool for early-stopping.

    Returns ``None`` when the canonical test labels are not available
    AND auto-download is disabled, so callers fall back to the custom
    500/63/63 partition.
    """
    test_img_dir = data_path / "test" / "img"
    test_ent_dir = data_path / "test" / "entities"
    if getattr(config, "canonical_sroie_enabled", False):
        from data.sroie_canonical import ensure_canonical_test_set
        try:
            ensure_canonical_test_set(config, data_path)
        except DataError as exc:
            logging.getLogger("kaggle2").warning(
                "canonical-SROIE auto-download failed (%s); falling back to "
                "the 500/63/63 internal split.", exc,
            )
            return None
    if not (test_img_dir.exists() and test_ent_dir.exists()):
        return None
    test_receipts = _load_receipts(test_img_dir, test_ent_dir)
    if not test_receipts:
        return None
    train_receipts = _load_receipts(
        data_path / "train" / "img", data_path / "train" / "entities",
    )
    if not train_receipts:
        return None
    # Bug 7 (extension to canonical-test boundary): the original Bug 7
    # guard in :func:`split_sroie` ensures val and test slices of the
    # 626-receipt training pool are disjoint.  When the canonical 347-
    # image test set replaces the internal 63-image test slice, a NEW
    # leakage surface appears at the train/val ↔ canonical-test boundary
    # because SROIE training and Task-3 test pools share a small number
    # of duplicate receipts post-correction (this is exactly what the
    # HF "v2" deduplication exists for).  Silent overlap would re-
    # introduce val≡test leakage on the canonical run.  Drop overlaps
    # and log loudly; never fail closed (vast.ai operators rerun this
    # path on every fresh instance).
    test_stems = {r.image_path.stem for r in test_receipts}
    overlap = [r for r in train_receipts if r.image_path.stem in test_stems]
    if overlap:
        logging.getLogger("kaggle2").warning(
            "canonical-SROIE: dropping %d train images that overlap test set "
            "(Bug 7 guard): %s", len(overlap),
            [r.image_path.stem for r in overlap[:5]],
        )
        train_receipts = [
            r for r in train_receipts if r.image_path.stem not in test_stems
        ]
    random.Random(config.seed).shuffle(train_receipts)
    val = train_receipts[:_N_VAL]
    train = train_receipts[_N_VAL:]
    return DataSplit(train=train, val=val, test=test_receipts)


def load_or_create_split(config: ExpConfig, data_path: Path) -> DataSplit:
    """Load cached split or create and persist (reproducibility across stages).

    Prefers the canonical 347-image SROIE test split when
    ``data_path/test/entities/`` is present; otherwise falls back to the
    custom 500/63/63 partition from the 626-image training set.
    """
    cache = Path(config.output_dir) / "split.json"
    seed = config.seed
    groups = ("train", "val", "test")

    # Attempt canonical test split first (prefers leaderboard protocol).
    canonical = _canonical_test_split(data_path, config)
    if canonical is not None:
        logging.getLogger("kaggle2").info(
            "Using canonical SROIE test split: %d train / %d val / %d test",
            len(canonical.train), len(canonical.val), len(canonical.test),
        )
        return canonical

    if cache.exists():
        raw = json.loads(cache.read_text())
        by_stem = {r.image_path.stem: r for r in _load_receipts(
            data_path / "train" / "img", data_path / "train" / "entities")}
        miss = [s for g in groups for s in raw[g] if s not in by_stem]
        if miss:
            raise DataError(f"Saved split missing images: {miss[:5]}...")
        return DataSplit(*([by_stem[s] for s in raw[g]] for g in groups))
    split = split_sroie(data_path, seed, bug_flags=config.bug_flags)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(
        {g: [r.image_path.stem for r in getattr(split, g)] for g in groups},
        indent=2,
    ))
    return split
