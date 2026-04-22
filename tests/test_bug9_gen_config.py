"""Regression tests for Bug 9: stale generation_config on reload.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: verify that _persist_generation_config re-pins the SROIE token IDs
    to disk after load_best_model_at_end restores a stale checkpoint, and
    that the assertion guard raises TrainError when the written file is wrong.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from core.errors import TrainError
from models._gen_config import _persist_generation_config

# ---------------------------------------------------------------------------

class _FakeGC:
    """Minimal generation_config stub with save_pretrained."""

    def __init__(self, out_dir: str, ids: dict[str, Any]) -> None:
        self._out_dir = out_dir
        for k, v in ids.items():
            setattr(self, k, v)

    def save_pretrained(self, path: str) -> None:
        gc_path = Path(path) / "generation_config.json"
        gc_path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {}
        for attr in (
            "decoder_start_token_id",
            "eos_token_id",
            "pad_token_id",
            "bos_token_id",
            "forced_bos_token_id",
            "forced_eos_token_id",
        ):
            data[attr] = getattr(self, attr, None)
        gc_path.write_text(json.dumps(data))


def _make_model(out_dir: str, stale_start_id: int = 0) -> Any:
    """Return a minimal model stub with stale generation_config."""
    gc = _FakeGC(out_dir, {
        "decoder_start_token_id": stale_start_id,
        "eos_token_id": 1,
        "pad_token_id": 1,
        "bos_token_id": stale_start_id,
        "forced_bos_token_id": 99,  # stale mBART default
        "forced_eos_token_id": 99,  # stale mBART default
    })
    config = SimpleNamespace(
        decoder_start_token_id=stale_start_id,
        eos_token_id=1,
        pad_token_id=1,
    )
    return SimpleNamespace(generation_config=gc, config=config)


# ---------------------------------------------------------------------------
# Tests for _persist_generation_config directly
# ---------------------------------------------------------------------------

def test_persist_writes_correct_ids(tmp_path: Path) -> None:
    """Helper writes the SROIE IDs and clears forced_* fields."""
    model = _make_model(str(tmp_path), stale_start_id=0)
    start_id, eos_id, pad_id = 50265, 50266, 1

    _persist_generation_config(model, str(tmp_path), start_id, eos_id, pad_id)

    data = json.loads((tmp_path / "generation_config.json").read_text())
    assert data["decoder_start_token_id"] == start_id
    assert data["eos_token_id"] == eos_id
    assert data["pad_token_id"] == pad_id
    assert data["bos_token_id"] == start_id
    assert data["forced_bos_token_id"] is None
    assert data["forced_eos_token_id"] is None


def test_persist_updates_model_config(tmp_path: Path) -> None:
    """Helper sets model.config as well as model.generation_config."""
    model = _make_model(str(tmp_path), stale_start_id=0)
    start_id, eos_id, pad_id = 50265, 50266, 1

    _persist_generation_config(model, str(tmp_path), start_id, eos_id, pad_id)

    assert model.config.decoder_start_token_id == start_id
    assert model.config.eos_token_id == eos_id
    assert model.config.pad_token_id == pad_id


def test_persist_survives_stale_reload_simulation(tmp_path: Path) -> None:
    """Simulate load_best_model_at_end mutating gc.decoder_start_token_id = 0."""
    model = _make_model(str(tmp_path), stale_start_id=50265)
    start_id, eos_id, pad_id = 50265, 50266, 1

    # Simulate the trainer reloading the best checkpoint and restoring stale ids
    model.generation_config.decoder_start_token_id = 0
    model.generation_config.forced_bos_token_id = 99

    _persist_generation_config(model, str(tmp_path), start_id, eos_id, pad_id)

    data = json.loads((tmp_path / "generation_config.json").read_text())
    assert data["decoder_start_token_id"] == start_id, (
        "Bug 9 regression: stale decoder_start_token_id survived re-pin"
    )
    assert data["forced_bos_token_id"] is None, (
        "Bug 9 regression: forced_bos_token_id not cleared"
    )


def test_persist_raises_train_error_when_disk_mismatch(tmp_path: Path) -> None:
    """Regression guard: patching gc.save_pretrained to write wrong IDs → TrainError."""

    class _BadGC(_FakeGC):
        def save_pretrained(self, path: str) -> None:
            # Writes wrong decoder_start_token_id (stale value)
            gc_path = Path(path) / "generation_config.json"
            gc_path.parent.mkdir(parents=True, exist_ok=True)
            gc_path.write_text(json.dumps({
                "decoder_start_token_id": 0,  # stale — wrong
                "eos_token_id": 50266,
                "pad_token_id": 1,
                "bos_token_id": 0,
                "forced_bos_token_id": None,
                "forced_eos_token_id": None,
            }))

    model = _make_model(str(tmp_path))
    model.generation_config = _BadGC(str(tmp_path), {
        "decoder_start_token_id": 0,
        "eos_token_id": 50266,
        "pad_token_id": 1,
        "bos_token_id": 0,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
    })

    with pytest.raises(TrainError, match="Bug-9 guard"):
        _persist_generation_config(model, str(tmp_path), 50265, 50266, 1)


def test_persist_noop_helper_raises_train_error(tmp_path: Path) -> None:
    """Regression: if re-pin is missing (gc writes stale IDs), TrainError is raised.

    Simulates the state after load_best_model_at_end if _persist_generation_config
    were a no-op: the gc on disk has the stale decoder_start_token_id (0 instead of
    the SROIE id), and the assertion guard fires on the next call that DOES run.
    """
    start_id, eos_id, pad_id = 50265, 50266, 1

    class _StaleGC(_FakeGC):
        """gc.save_pretrained writes stale IDs, simulating no-re-pin path."""

        def save_pretrained(self, path: str) -> None:
            gc_path = Path(path) / "generation_config.json"
            gc_path.parent.mkdir(parents=True, exist_ok=True)
            gc_path.write_text(json.dumps({
                "decoder_start_token_id": 0,  # stale — not re-pinned
                "eos_token_id": eos_id,
                "pad_token_id": pad_id,
                "bos_token_id": 0,
                "forced_bos_token_id": 99,
                "forced_eos_token_id": 99,
            }))

    model = _make_model(str(tmp_path))
    model.generation_config = _StaleGC(str(tmp_path), {
        "decoder_start_token_id": 0,
        "eos_token_id": eos_id,
        "pad_token_id": pad_id,
        "bos_token_id": 0,
        "forced_bos_token_id": 99,
        "forced_eos_token_id": 99,
    })

    with pytest.raises(TrainError, match="Bug-9 guard"):
        _persist_generation_config(model, str(tmp_path), start_id, eos_id, pad_id)


def test_persist_clears_forced_eos(tmp_path: Path) -> None:
    """forced_eos_token_id is also cleared (mBART leaks both forced IDs)."""
    model = _make_model(str(tmp_path))
    model.generation_config.forced_eos_token_id = 2  # stale mBART default

    _persist_generation_config(model, str(tmp_path), 50265, 50266, 1)

    data = json.loads((tmp_path / "generation_config.json").read_text())
    assert data["forced_eos_token_id"] is None
