"""Tests for _write_eval_diag: verifies donut_eval_diag.json is written correctly."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from models.donut_eval import _write_eval_diag

_EXPECTED_TOP_LEVEL_KEYS = {
    "image_processor_size",
    "decoder_start_token_id",
    "eos_token_id",
    "pad_token_id",
    "tokenizer_has_s_sroie",
    "tokenizer_s_sroie_id",
    "tokenizer_eos_s_sroie_id",
    "model_config_vocab_size",
    "tokenizer_vocab_size",
    "lm_head_out_features",
    "generation_config",
    "samples",
}

_EXPECTED_GENERATION_CONFIG_KEYS = {
    "forced_bos_token_id",
    "forced_eos_token_id",
    "decoder_start_token_id",
    "eos_token_id",
    "bos_token_id",
}


def _make_mock_processor() -> MagicMock:
    proc = MagicMock()
    proc.image_processor.size = {"height": 960, "width": 1280}
    proc.tokenizer.convert_tokens_to_ids.return_value = [42]
    proc.tokenizer.unk_token_id = 0
    proc.tokenizer.__len__ = lambda self: 57522  # type: ignore[assignment]
    return proc


def _make_mock_model() -> MagicMock:
    model = MagicMock()
    model.config.pad_token_id = 1
    model.config.decoder.vocab_size = 57522
    model.decoder.lm_head.weight.shape = (57522, 1024)
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None
    model.generation_config.decoder_start_token_id = 42
    model.generation_config.eos_token_id = 43
    model.generation_config.bos_token_id = 42
    return model


def test_diag_file_is_created(tmp_path: Path) -> None:
    """_write_eval_diag must create donut_eval_diag.json in out_dir."""
    _write_eval_diag(_make_mock_processor(), _make_mock_model(), 42, 43, [], str(tmp_path))
    assert (tmp_path / "donut_eval_diag.json").exists()


def test_diag_has_all_required_top_level_keys(tmp_path: Path) -> None:
    """donut_eval_diag.json must contain every key in the schema."""
    _write_eval_diag(_make_mock_processor(), _make_mock_model(), 42, 43, [], str(tmp_path))
    data = json.loads((tmp_path / "donut_eval_diag.json").read_text())
    missing = _EXPECTED_TOP_LEVEL_KEYS - set(data.keys())
    assert not missing, f"Missing top-level keys: {sorted(missing)}"


def test_diag_generation_config_keys(tmp_path: Path) -> None:
    """generation_config sub-dict must contain all five expected keys."""
    _write_eval_diag(_make_mock_processor(), _make_mock_model(), 42, 43, [], str(tmp_path))
    data = json.loads((tmp_path / "donut_eval_diag.json").read_text())
    gc = data["generation_config"]
    missing = _EXPECTED_GENERATION_CONFIG_KEYS - set(gc.keys())
    assert not missing, f"Missing generation_config keys: {sorted(missing)}"


def test_diag_lm_head_out_features_correct(tmp_path: Path) -> None:
    """lm_head_out_features must equal lm_head.weight.shape[0]."""
    _write_eval_diag(_make_mock_processor(), _make_mock_model(), 42, 43, [], str(tmp_path))
    data = json.loads((tmp_path / "donut_eval_diag.json").read_text())
    assert data["lm_head_out_features"] == 57522


def test_diag_samples_stored(tmp_path: Path) -> None:
    """Provided sample dicts must appear verbatim in donut_eval_diag.json."""
    samples = [
        {"image_id": "X00016469612", "tokens_full": "<s_sroie><s_company>A</s_company></s_sroie>",
         "raw_token2json": {"company": "A"}, "parsed": {"company": "A"}, "gt": {"company": "A"}},
    ]
    _write_eval_diag(_make_mock_processor(), _make_mock_model(), 42, 43, samples, str(tmp_path))
    data = json.loads((tmp_path / "donut_eval_diag.json").read_text())
    assert len(data["samples"]) == 1
    assert data["samples"][0]["image_id"] == "X00016469612"
    assert data["samples"][0]["parsed"] == {"company": "A"}


def test_diag_tokenizer_s_sroie_id_set(tmp_path: Path) -> None:
    """tokenizer_s_sroie_id must be the first element returned by convert_tokens_to_ids."""
    _write_eval_diag(_make_mock_processor(), _make_mock_model(), 42, 43, [], str(tmp_path))
    data = json.loads((tmp_path / "donut_eval_diag.json").read_text())
    assert data["tokenizer_s_sroie_id"] == 42


def test_diag_decoder_start_token_id_matches_arg(tmp_path: Path) -> None:
    """decoder_start_token_id must match the start_id argument passed in."""
    _write_eval_diag(_make_mock_processor(), _make_mock_model(), 99, 100, [], str(tmp_path))
    data = json.loads((tmp_path / "donut_eval_diag.json").read_text())
    assert data["decoder_start_token_id"] == 99
    assert data["eos_token_id"] == 100
