"""Unit tests for core.config — validation and Bug 6 floor."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import load_config


def _minimal_config() -> dict[str, object]:
    return {
        "seed": 42,
        "base_model": "naver-clova-ix/donut-base",
        "trocr_model": "microsoft/trocr-base-printed",
        "yolo_model": "yolov8n.pt",
        "image_size": [1280, 960],
        "yolo_img_size": 512,
        "max_length": 768,
        "trocr_max_len": 128,
        "epochs_donut": 10,
        "epochs_yolo": 50,
        "epochs_trocr": 10,
        "epochs_assigner": 20,
        "batch_size": 8,
        "grad_accum": 2,
        "lr": 5e-5,
        "lr_decoder": 1e-4,
        "warmup_steps": 40,
        "weight_decay": 0.01,
        "label_smoothing": 0.1,
        "precision": "bf16",
        "patience": 3,
        "max_grad_norm": 1.0,
        "fields": ["company", "date", "address", "total"],
        "new_tokens": ["<s_sroie>", "</s_sroie>"],
        "sroie_url": "https://example.com/sroie.git",
        "data_dir": "./data/sroie_cache",
        "output_dir": "./results",
        "paper_template": "./report/template.tex",
        "paper_output": "./report/paper_filled.tex",
    }


def _write(tmp_path: Path, cfg: dict[str, object]) -> str:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return str(p)


def test_load_minimal_config_returns_expconfig(tmp_path: Path) -> None:
    cfg = load_config(_write(tmp_path, _minimal_config()))
    assert cfg.seed == 42
    assert cfg.image_size == (1280, 960)
    assert cfg.fields == ["company", "date", "address", "total"]


def test_missing_required_key_raises(tmp_path: Path) -> None:
    raw = _minimal_config()
    del raw["seed"]
    with pytest.raises(ValueError, match="missing required keys"):
        load_config(_write(tmp_path, raw))


def test_trocr_epoch_floor_enforced(tmp_path: Path) -> None:
    # Bug 6: TrOCR undertrained → all-empty outputs → pipeline F1=0.
    raw = _minimal_config()
    raw["epochs_trocr"] = 2
    with pytest.raises(ValueError, match="TrOCR will underfit"):
        load_config(_write(tmp_path, raw))


def test_unknown_keys_preserved_in_extra(tmp_path: Path) -> None:
    raw = _minimal_config()
    raw["custom_tag"] = "experimental"
    cfg = load_config(_write(tmp_path, raw))
    assert cfg.extra.get("custom_tag") == "experimental"
