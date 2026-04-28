"""Shared pytest fixtures + helpers for kaggle2.

Provides:
* :func:`write_min_config` — writes a minimal valid ``config.json`` to
  a tmp dir and returns its path; used by per-PR tests that exercise
  :func:`core.config.load_config`.
* ``min_config_path`` fixture — same thing as a fixture for tests that
  prefer the dependency-injection style.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure ``from conftest import write_min_config`` resolves when tests
# are collected from the ``tests/`` directory.  pytest does not put
# the tests dir on ``sys.path`` automatically without ``rootdir`` or
# ``--rootdir`` config; doing it here is robust.
_TESTS_DIR = Path(__file__).resolve().parent
if str(_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_TESTS_DIR))

_MIN_CONFIG: dict[str, object] = {
    "seed": 42,
    "base_model": "naver-clova-ix/donut-base",
    "trocr_model": "microsoft/trocr-base-handwritten",
    "yolo_model": "yolov8n.pt",
    "image_size": [1280, 960],
    "yolo_img_size": 1280,
    "max_length": 768,
    "trocr_max_len": 64,
    "epochs_donut": 1,
    "epochs_yolo": 1,
    "epochs_trocr": 5,
    "epochs_assigner": 1,
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
    "paper_template": str(
        (Path(__file__).parent.parent / "report" / "template.tex").resolve(),
    ),
    "paper_output": "./report/paper_filled.tex",
}


def write_min_config(tmp_path: Path, **overrides: object) -> Path:
    """Create a minimal ``config.json`` under ``tmp_path``."""
    cfg = dict(_MIN_CONFIG)
    cfg.setdefault("data_dir", str(tmp_path))
    cfg.setdefault("output_dir", str(tmp_path))
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


@pytest.fixture
def min_config_path(tmp_path: Path) -> Path:
    """A minimal valid ``config.json`` written under a per-test tmp dir."""
    return write_min_config(tmp_path)
