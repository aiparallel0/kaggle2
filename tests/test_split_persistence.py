"""Unit tests for split persistence — prevents train/eval drift."""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from data.sroie import load_or_create_split


def _stub_sroie(root: Path, n: int = 130) -> Path:
    img_dir = root / "train" / "img"
    ent_dir = root / "train" / "entities"
    img_dir.mkdir(parents=True, exist_ok=True)
    ent_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        p = img_dir / f"r{i:04d}.jpg"
        Image.new("RGB", (32, 32), "white").save(p)
        (ent_dir / f"r{i:04d}.json").write_text(
            json.dumps({"company": "A", "date": "2020", "address": "x", "total": "1"}),
        )
    return root


def test_load_or_create_split_creates_and_reuses(tmp_path: Path) -> None:
    data_path = _stub_sroie(tmp_path / "data")
    cache = tmp_path / "split.json"
    s1 = load_or_create_split(data_path, seed=42, cache=cache)
    assert cache.exists()
    s2 = load_or_create_split(data_path, seed=42, cache=cache)
    assert [r.image_path.stem for r in s1.test] == [r.image_path.stem for r in s2.test]
    assert [r.image_path.stem for r in s1.val] == [r.image_path.stem for r in s2.val]


def test_saved_split_survives_reseed(tmp_path: Path) -> None:
    # Once persisted, a different seed must not change the returned split.
    data_path = _stub_sroie(tmp_path / "data")
    cache = tmp_path / "split.json"
    s1 = load_or_create_split(data_path, seed=42, cache=cache)
    s2 = load_or_create_split(data_path, seed=999, cache=cache)
    assert [r.image_path.stem for r in s1.test] == [r.image_path.stem for r in s2.test]


def test_split_has_no_val_test_overlap(tmp_path: Path) -> None:
    data_path = _stub_sroie(tmp_path / "data")
    s = load_or_create_split(data_path, seed=7, cache=tmp_path / "split.json")
    val = {r.image_path.stem for r in s.val}
    test = {r.image_path.stem for r in s.test}
    assert not (val & test)
