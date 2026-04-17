"""Smoke test: every module imports cleanly, ExpConfig loads from config.json.

Guards against the class of regression where a copilot-authored refactor
silently breaks an import path that only manifests on vast.ai during
`make all`, long after `make check` (mypy/ruff) has already passed.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from core.config import load_config

_MODULES = (
    "core.config", "core.errors", "core.metrics", "core.seed", "core.types",
    "data.sroie",
    "models.attention_assign", "models.assigner_train",
    "models.donut_eval", "models.donut_train",
    "models.pipeline_eval", "models.rule_based",
    "models.trocr_train", "models.yolo_train",
    "report.inject",
)


@pytest.mark.parametrize("name", _MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_load_project_config_json() -> None:
    root = Path(__file__).resolve().parent.parent
    cfg_path = root / "config.json"
    assert cfg_path.exists(), f"config.json missing at {cfg_path}"
    cfg = load_config(str(cfg_path))
    assert cfg.epochs_trocr >= 5  # Bug 6 floor
    assert cfg.epochs_donut >= 1
    assert cfg.fields, "fields must be non-empty"
    assert cfg.paper_template.endswith(".tex")
    assert cfg.paper_output.endswith(".tex")
