"""Tests for Bug 14: canonical-SROIE GT entity-format poisoning guards.

Validates that:
- _flatten_json_entities is path-scoped AND content-scoped (rejects box files).
- _parse_entities_txt handles JSON-first and rejects Task-1/2 box-file lines.
- _canonical_test_split raises DataError when entities/ holds box files.

All tests are CPU-only; no network access; no torch.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config import load_config
from core.errors import DataError
from data.sroie import _canonical_test_split, _parse_entities_txt
from data.sroie_canonical import _flatten_json_entities, _is_task3_entity_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_VALID_JSON_GT = json.dumps(
    {"company": "ACME SDN BHD", "date": "01/01/2020",
     "address": "1 JALAN TEST", "total": "10.00"}
)
_BOX_FILE_CONTENT = (
    "98,26,321,26,321,66,98,66,TAN CHAY YEE\n"
    "129,142,287,142,287,160,129,160,roc no: 538358-H\n"
    "69,226,339,226,339,248,69,248,tel: 07-388 2218\n"
)


def _write_min_config(tmp_path: Path, **overrides: object) -> Path:
    cfg: dict[str, object] = {
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
        "data_dir": str(tmp_path / "data"),
        "output_dir": str(tmp_path / "results"),
        "paper_template": str(
            (Path(__file__).parent.parent / "report" / "template.tex").resolve()
        ),
        "paper_output": "./report/paper_filled.tex",
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    return p


# ---------------------------------------------------------------------------
# _flatten_json_entities tests
# ---------------------------------------------------------------------------

def test_flatten_json_entities_rejects_box_files(tmp_path: Path) -> None:
    """JSON entity in test/entities/ is accepted; box file with same basename
    in task1_2_test/box/ is silently skipped -- not overwritten."""
    mirror = tmp_path / "mirror"
    # Valid Task-3 JSON entity under the correct path
    ent_dir = mirror / "test" / "entities"
    ent_dir.mkdir(parents=True)
    (ent_dir / "X1.txt").write_text(_VALID_JSON_GT)
    # Box file with SAME basename under a Task-1/2 path
    box_dir = mirror / "task1_2_test" / "box"
    box_dir.mkdir(parents=True)
    (box_dir / "X1.txt").write_text(_BOX_FILE_CONTENT)

    dst = tmp_path / "dst"
    n = _flatten_json_entities(mirror, dst)

    assert n == 1, f"Expected 1 entity placed, got {n}"
    result = (dst / "X1.txt").read_text()
    assert result == _VALID_JSON_GT, "Destination file should be the JSON GT, not the box content"


def test_flatten_json_entities_rejects_non_json_in_entities_dir(tmp_path: Path) -> None:
    """A .txt file in test/entities/ whose first byte is NOT '{' is skipped."""
    mirror = tmp_path / "mirror"
    ent_dir = mirror / "test" / "entities"
    ent_dir.mkdir(parents=True)
    (ent_dir / "Y.txt").write_text("roc no: 538358-H\ndate: 01/01/2020\n")

    dst = tmp_path / "dst"
    n = _flatten_json_entities(mirror, dst)

    assert n == 0, "Non-JSON entity file must be skipped"
    assert not (dst / "Y.txt").exists()


# ---------------------------------------------------------------------------
# _is_task3_entity_path tests
# ---------------------------------------------------------------------------

def test_is_task3_entity_path_accepts_canonical_paths(tmp_path: Path) -> None:
    for subdir in (("test", "entities"), ("test", "key"), ("data", "key")):
        p = tmp_path.joinpath(*subdir) / "X.txt"
        assert _is_task3_entity_path(p), f"Should accept {subdir}"


def test_is_task3_entity_path_rejects_box_paths(tmp_path: Path) -> None:
    for subdir in (("task1_2_test", "box"), ("test", "box"), ("train", "entities")):
        p = tmp_path.joinpath(*subdir) / "X.txt"
        assert not _is_task3_entity_path(p), f"Should reject {subdir}"


# ---------------------------------------------------------------------------
# _parse_entities_txt tests
# ---------------------------------------------------------------------------

def test_parse_entities_json_first(tmp_path: Path) -> None:
    """When text is a JSON dict, keys are lowercased and values preserved."""
    text = json.dumps({"Company": "ACME SDN BHD", "Date": "01/01/2020",
                       "Address": "1 JALAN TEST", "Total": "10.00"})
    result = _parse_entities_txt(text)
    assert result["company"] == "ACME SDN BHD"
    assert result["date"] == "01/01/2020"
    assert result["address"] == "1 JALAN TEST"
    assert result["total"] == "10.00"


def test_parse_entities_rejects_box_lines(tmp_path: Path) -> None:
    """Box-file content (coord-prefixed keys) must all be rejected → empty dict."""
    result = _parse_entities_txt(_BOX_FILE_CONTENT)
    assert result == {}, f"Expected empty dict, got {result!r}"


def test_parse_entities_legacy_colon_format(tmp_path: Path) -> None:
    """Legacy key:value lines (no coord prefix) are still parsed correctly."""
    text = "company: MY COMPANY\ndate: 12/12/2021\ntotal: 99.00\n"
    result = _parse_entities_txt(text)
    assert result["company"] == "MY COMPANY"
    assert result["date"] == "12/12/2021"
    assert result["total"] == "99.00"


def test_parse_entities_preserves_colons_in_values(tmp_path: Path) -> None:
    """Values containing colons (e.g. time) are preserved intact."""
    text = "time: 12:30:00\n"
    result = _parse_entities_txt(text)
    assert result["time"] == "12:30:00"


# ---------------------------------------------------------------------------
# _canonical_test_split raises on poisoned entities
# ---------------------------------------------------------------------------

def test_canonical_split_raises_on_poisoned_entities(tmp_path: Path) -> None:
    """_canonical_test_split must raise DataError (mentioning box files / JSON GT)
    when entities/ contains only Task-1 box files, not Task-3 JSON GT."""
    # Set up test/img + test/entities (poisoned with box file)
    (tmp_path / "test" / "img").mkdir(parents=True)
    (tmp_path / "test" / "img" / "X0.jpg").write_bytes(b"\xff\xd8\xff\xe0")
    (tmp_path / "test" / "entities").mkdir(parents=True)
    (tmp_path / "test" / "entities" / "X0.txt").write_text(_BOX_FILE_CONTENT)

    cfg_path = _write_min_config(
        tmp_path, canonical_sroie_enabled=False, data_dir=str(tmp_path),
    )
    config = load_config(str(cfg_path))

    with pytest.raises(DataError, match="Task-1 box files|JSON GT"):
        _canonical_test_split(tmp_path, config)
