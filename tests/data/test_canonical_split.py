"""Tests for the canonical SROIE 347-image test pathway and paper-variant switch."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from core.config import load_config
from core.errors import DataError
from data.sroie_canonical import (
    _verify_sha256,
    ensure_canonical_test_set,
)


def _write_min_config(tmp_path: Path, **overrides: object) -> Path:
    """Materialise a minimal config.json for load_config exercise tests."""
    cfg: dict[str, object] = {
        "seed": 42,
        "base_model": "naver-clova-ix/donut-base",
        "trocr_model": "microsoft/trocr-base-handwritten",
        "yolo_model": "yolov8n.pt",
        "image_size": [1280, 960],
        "yolo_image_size": 1280,
        "max_length": 768,
        "trocr_max_len": 64,
        "epochs_donut": 1,
        "epochs_yolo": 1,
        "epochs_trocr": 5,
        "epochs_focus": 1,
        "batch_size": 1,
        "grad_accum": 1,
        "lr": 1e-5,
        "lr_decoder": 1e-4,
        "warmup_steps": 50,
        "weight_decay": 0.01,
        "label_smoothing": 0.1,
        "precision": "bf16",
        "patience": 3,
        "max_grad_norm": 1.0,
        "fields": ["company", "date", "address", "total"],
        "new_tokens": ["<s_sroie>"],
        "sroie_url": "https://example.invalid/sroie.git",
        "data_dir": "./data/sroie_cache",
        "output_dir": "./results",
        "paper_template": str(
            (Path(__file__).parents[2] / "report" / "template.tex").resolve()
        ),
        "paper_output": "./report/paper_filled.tex",
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


def test_sha256_mismatch_raises(tmp_path: Path) -> None:
    """A digest mismatch must raise DataError before any extraction."""
    fake = tmp_path / "fake.zip"
    fake.write_bytes(b"not the real archive")
    with pytest.raises(DataError, match="sha256 mismatch"):
        _verify_sha256(fake, "a" * 64, "some mirror")


def test_sha256_none_skips(tmp_path: Path) -> None:
    """Expected=None skips verification (used for unpinned RRC primary)."""
    fake = tmp_path / "fake.zip"
    fake.write_bytes(b"any content")
    _verify_sha256(fake, None, "primary RRC")  # must not raise


def test_canonical_idempotent_when_already_extracted(tmp_path: Path) -> None:
    """Pre-populating test/img/ and test/entities/ with exactly 347 files short-circuits."""
    img_dir = tmp_path / "test" / "img"
    ent_dir = tmp_path / "test" / "entities"
    img_dir.mkdir(parents=True)
    ent_dir.mkdir(parents=True)
    for i in range(347):
        (img_dir / f"X{i:05d}.jpg").write_bytes(b"\xff\xd8\xff\xe0")  # JPEG SOI
        (ent_dir / f"X{i:05d}.txt").write_bytes(b'{"company":"F","date":"d","address":"a","total":"t"}')
    cfg_path = _write_min_config(
        tmp_path, canonical_sroie_enabled=True, data_dir=str(tmp_path),
    )
    config = load_config(str(cfg_path))
    # Must return without attempting any network I/O.
    out = ensure_canonical_test_set(config, tmp_path)
    # API: ensure_canonical_test_set now returns a CanonicalStatus
    # describing the outcome (mirror_used, counts, fallback flag).
    assert out.mirror_used == "cached"
    assert out.n_img_collected == 347
    assert out.fallback_triggered is False
    assert len(list(img_dir.glob("*.jpg"))) == 347


def test_paper_variant_advanced_picks_advanced_template(tmp_path: Path) -> None:
    """paper_variant=advanced selects template_focus.tex when present."""
    cfg_path = _write_min_config(tmp_path, paper_variant="focus")
    os.environ.pop("KAGGLE2_PAPER_VARIANT", None)
    config = load_config(str(cfg_path))
    assert config.paper_template.endswith("template_focus.tex"), (
        f"expected template_focus.tex, got {config.paper_template}"
    )


def test_paper_variant_basic_picks_basic_template(tmp_path: Path) -> None:
    """paper_variant=basic selects template_baseline.tex when present."""
    cfg_path = _write_min_config(tmp_path, paper_variant="baseline")
    os.environ.pop("KAGGLE2_PAPER_VARIANT", None)
    config = load_config(str(cfg_path))
    assert config.paper_template.endswith("template_baseline.tex"), (
        f"expected template_baseline.tex, got {config.paper_template}"
    )


def test_paper_variant_env_override_wins(tmp_path: Path) -> None:
    """KAGGLE2_PAPER_VARIANT env var beats config.json paper_variant."""
    cfg_path = _write_min_config(tmp_path, paper_variant="focus")
    os.environ["KAGGLE2_PAPER_VARIANT"] = "baseline"
    try:
        config = load_config(str(cfg_path))
        assert config.paper_template.endswith("template_baseline.tex")
    finally:
        os.environ.pop("KAGGLE2_PAPER_VARIANT", None)


def test_strip_gtocr_keys_helper() -> None:
    """_strip_gtocr_keys removes every gtocr/rulebased/oracle prefix."""
    from stages.eval import _strip_gtocr_keys
    d: dict[str, object] = {
        "donut_f1": 0.8,
        "pipeline_f1": 0.7,
        "gtocr_rulebased_f1": 0.5,
        "gtocr_rulebased_em": 0.3,
        "rulebased_f1_company": 0.2,
        "oracle_patch_f1_if_applied": 0.4,
        "n_trials": 1,
    }
    _strip_gtocr_keys(d)
    assert "donut_f1" in d and "pipeline_f1" in d and "n_trials" in d
    assert not any(k.startswith(("gtocr_", "rulebased_", "oracle_patch_")) for k in d)


def test_canonical_config_keys_round_trip(tmp_path: Path) -> None:
    """The new canonical_sroie_* + paper_variant keys parse via load_config."""
    cfg_path = _write_min_config(
        tmp_path,
        canonical_sroie_enabled=True,
        canonical_sroie_test_url="https://example.invalid/img.zip",
        canonical_sroie_gt_url="https://example.invalid/gt.zip",
        canonical_sroie_hf_repo="Metric-AI/icdar_sroie",
        canonical_sroie_hf_revision="main",
        paper_variant="focus",
    )
    os.environ.pop("KAGGLE2_PAPER_VARIANT", None)
    config = load_config(str(cfg_path))
    assert config.canonical_sroie_enabled is True
    assert config.canonical_sroie_test_url == "https://example.invalid/img.zip"
    assert config.canonical_sroie_gt_url == "https://example.invalid/gt.zip"
    assert config.canonical_sroie_hf_repo == "Metric-AI/icdar_sroie"
    assert config.canonical_sroie_hf_revision == "main"
    assert config.paper_variant == "focus"
