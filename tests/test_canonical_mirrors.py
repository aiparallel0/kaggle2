"""Per-mirror integration tests for canonical-SROIE MirrorAdapters.

Builds in-memory fixture archives that mimic each mirror's on-disk layout
(RRC dual-zip) and asserts placed-file counts come out at exactly 347/347.

All HuggingFace calls are mocked via monkeypatch on
``data.sroie_canonical_hf.try_huggingface`` — no real network access.
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import data.sroie_canonical as sc
import data.sroie_canonical_hf as sc_hf
from core.config import load_config
from core.errors import DataError
from data.sroie_canonical import (
    _TASK3_TEST_COUNT,
    CanonicalStatus,
    MirrorAdapter,
    ensure_canonical_test_set,
)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #

_JPEG_SOI = b"\xff\xd8\xff\xe0fakejpeg"
_GT_JSON = b'{"company": "FOO", "date": "01/01/2024", "address": "X", "total": "1.00"}'

# Canonical stems used in all tests (X00000 … X00346)
_CANON_STEMS = [f"X{i:05d}" for i in range(_TASK3_TEST_COUNT)]


# --------------------------------------------------------------------------- #
# Fixture builders — RRC                                                       #
# --------------------------------------------------------------------------- #

def _build_rrc_archives(workdir: Path) -> tuple[Path, Path]:
    """Mimic the RRC dual-zip layout: images + GT in two separate archives."""
    img_zip = workdir / "rrc_img.zip"
    gt_zip = workdir / "rrc_gt.zip"
    with zipfile.ZipFile(img_zip, "w") as zf:
        for stem in _CANON_STEMS:
            zf.writestr(f"task3-test(347p)/{stem}.jpg", _JPEG_SOI)
    with zipfile.ZipFile(gt_zip, "w") as zf:
        for stem in _CANON_STEMS:
            zf.writestr(f"test/entities/{stem}.txt", _GT_JSON)
    return img_zip, gt_zip


# --------------------------------------------------------------------------- #
# Fixture builders — HuggingFace mock                                          #
# --------------------------------------------------------------------------- #

def _make_hf_fixture(
    *,
    n_files: int = _TASK3_TEST_COUNT,
    stems: list[str] | None = None,
    gt_overrides: dict[str, str] | None = None,
) -> Callable[[Path, str, str], Path | None]:
    """Return a ``try_huggingface`` stub that materialises jpg+json under workdir/hf.

    Parameters
    ----------
    n_files:
        How many files to write (default 347).
    stems:
        Explicit stem list; defaults to canonical ``_CANON_STEMS[:n_files]``.
    gt_overrides:
        Fields to override in the written GT JSON (e.g. ``{"company": "BAR"}``).
    """
    effective_stems = stems if stems is not None else _CANON_STEMS[:n_files]

    def _stub(workdir: Path, repo_id: str, revision: str) -> Path | None:
        work_hf = workdir / "hf"
        img_dir = work_hf / "img"
        ent_dir = work_hf / "entities"
        img_dir.mkdir(parents=True, exist_ok=True)
        ent_dir.mkdir(parents=True, exist_ok=True)
        for stem in effective_stems:
            (img_dir / f"{stem}.jpg").write_bytes(_JPEG_SOI)
            gt: dict[str, str] = {
                "company": "FOO", "date": "01/01/2024",
                "address": "X", "total": "1.00",
            }
            if gt_overrides:
                gt.update(gt_overrides)
            (ent_dir / f"{stem}.json").write_text(json.dumps(gt), encoding="utf-8")
        return work_hf

    return _stub


def _make_hf_returning_none() -> Callable[[Path, str, str], Path | None]:
    """Stub that returns None (simulates HF unavailable / download failure)."""
    def _stub(workdir: Path, repo_id: str, revision: str) -> Path | None:
        return None
    return _stub


# --------------------------------------------------------------------------- #
# Config helper                                                                #
# --------------------------------------------------------------------------- #

def _min_config(tmp_path: Path, **overrides: Any) -> Any:
    cfg: dict[str, Any] = {
        "seed": 42,
        "base_model": "x", "trocr_model": "x", "yolo_model": "x",
        "image_size": [1280, 960], "yolo_img_size": 1280,
        "max_length": 768, "trocr_max_len": 64,
        "epochs_donut": 1, "epochs_yolo": 1, "epochs_trocr": 5,
        "epochs_assigner": 1,
        "batch_size": 1, "grad_accum": 1, "lr": 1e-5, "lr_decoder": 1e-4,
        "warmup_steps": 50, "weight_decay": 0.01, "label_smoothing": 0.1,
        "precision": "bf16", "patience": 3, "max_grad_norm": 1.0,
        "fields": ["company", "date", "address", "total"],
        "new_tokens": ["<s_sroie>"],
        "sroie_url": "https://example.invalid/sroie.git",
        "data_dir": str(tmp_path / "sroie_cache"),
        "output_dir": str(tmp_path / "out"),
        "paper_template": str(
            (Path(__file__).parent.parent / "report" / "template.tex").resolve(),
        ),
        "paper_output": str(tmp_path / "paper.tex"),
        "canonical_sroie_enabled": True,
        "canonical_sroie_test_url": "https://test.invalid/img.zip",
        "canonical_sroie_gt_url": "https://test.invalid/gt.zip",
        "canonical_sroie_hf_repo": "Metric-AI/icdar_sroie",
        "canonical_sroie_hf_revision": "main",
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    os.environ.pop("KAGGLE2_PAPER_VARIANT", None)
    return load_config(str(p))


def _patch_downloads(
    monkeypatch: pytest.MonkeyPatch,
    fakes: dict[str, Path],
) -> None:
    """Make ``_download`` copy from ``fakes[url]`` instead of hitting network."""
    import shutil

    def fake_download(url: str, dst: Path, timeout: float = 60.0) -> None:
        if url not in fakes:
            raise DataError(f"fixture: no fake for {url}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fakes[url], dst)

    monkeypatch.setattr(sc, "_download", fake_download)


# --------------------------------------------------------------------------- #
# RRC primary tests (kept from original suite)                                 #
# --------------------------------------------------------------------------- #

def test_rrc_adapter_collects_exactly_347(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RRC dual-zip layout → exactly 347 img + 347 ent placed."""
    workdir = tmp_path / "src"
    workdir.mkdir()
    img_zip, gt_zip = _build_rrc_archives(workdir)
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {
        config.canonical_sroie_test_url: img_zip,
        config.canonical_sroie_gt_url: gt_zip,
    })
    data_path = Path(config.data_dir)
    status = ensure_canonical_test_set(config, data_path)
    assert status.mirror_used == "rrc"
    assert status.n_img_collected == _TASK3_TEST_COUNT
    assert status.n_ent_collected == _TASK3_TEST_COUNT
    assert status.fallback_triggered is False
    assert len(list((data_path / "test" / "img").glob("*.jpg"))) == _TASK3_TEST_COUNT
    assert len(list((data_path / "test" / "entities").glob("*.txt"))) == _TASK3_TEST_COUNT


def test_rrc_adapter_records_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RRC success path populates stems_sha256 and gt_content_sha256 in status."""
    workdir = tmp_path / "src"
    workdir.mkdir()
    img_zip, gt_zip = _build_rrc_archives(workdir)
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {
        config.canonical_sroie_test_url: img_zip,
        config.canonical_sroie_gt_url: gt_zip,
    })
    status = ensure_canonical_test_set(config, Path(config.data_dir))
    assert status.stems_sha256 != ""
    assert status.gt_content_sha256 != ""


# --------------------------------------------------------------------------- #
# HuggingFace fallback tests (B7)                                              #
# --------------------------------------------------------------------------- #

def test_hf_fallback_succeeds_when_primary_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RRC URLs → DataError; HF fixture → 347/347; mirror_used == 'huggingface'."""
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})  # no RRC fakes → primary fails
    monkeypatch.setattr(sc_hf, "try_huggingface", _make_hf_fixture())
    data_path = Path(config.data_dir)
    status = ensure_canonical_test_set(config, data_path)
    assert status.mirror_used == "huggingface"
    assert status.n_img_collected == _TASK3_TEST_COUNT
    assert status.n_ent_collected == _TASK3_TEST_COUNT
    assert status.fallback_triggered is False
    assert status.stems_sha256 != ""
    assert status.gt_content_sha256 != ""
    assert len(list((data_path / "test" / "img").glob("*.jpg"))) == _TASK3_TEST_COUNT


def test_hf_stem_set_drift_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned stems sha + HF with one swapped stem → DataError('stem-set drift')."""
    # Pin to a non-matching SHA (pattern from C2)
    monkeypatch.setattr(sc, "_RRC_TASK3_STEMS_SHA256", "deadbeef" * 8)
    # HF returns 347 files but one stem is wrong
    wrong_stems = _CANON_STEMS[:-1] + ["WRONGSTEM"]
    monkeypatch.setattr(sc_hf, "try_huggingface", _make_hf_fixture(stems=wrong_stems))
    monkeypatch.setattr(sc, "_try_primary", lambda *a: None)
    with pytest.raises(DataError, match="stem-set drift"):
        ensure_canonical_test_set(_min_config(tmp_path), Path(tmp_path / "sroie_cache"))


def test_hf_gt_content_drift_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pinned GT sha + HF with mutated GT value → DataError('GT content drift')."""
    # Pin to a non-matching SHA
    monkeypatch.setattr(sc, "_RRC_TASK3_GT_SHA256", "deadbeef" * 8)
    # HF returns canonical stems but with a mutated company field
    monkeypatch.setattr(
        sc_hf, "try_huggingface",
        _make_hf_fixture(gt_overrides={"company": "MUTATED"}),
    )
    monkeypatch.setattr(sc, "_try_primary", lambda *a: None)
    with pytest.raises(DataError, match="GT content drift"):
        ensure_canonical_test_set(_min_config(tmp_path), Path(tmp_path / "sroie_cache"))


def test_first_run_records_hashes_in_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PIN_ON_FIRST_RUN (both pins) → success; sidecar has non-empty hashes."""
    assert sc._RRC_TASK3_STEMS_SHA256 == "PIN_ON_FIRST_RUN"
    assert sc._RRC_TASK3_GT_SHA256 == "PIN_ON_FIRST_RUN"
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})
    monkeypatch.setattr(sc_hf, "try_huggingface", _make_hf_fixture())
    # Also pre-populate train/ so _canonical_test_split can build a split
    train_img = Path(config.data_dir) / "train" / "img"
    train_ent = Path(config.data_dir) / "train" / "entities"
    train_img.mkdir(parents=True)
    train_ent.mkdir(parents=True)
    for i in range(80):
        (train_img / f"T{i:05d}.jpg").write_bytes(_JPEG_SOI)
        (train_ent / f"T{i:05d}.txt").write_bytes(_GT_JSON)
    from data.sroie import _canonical_test_split
    split = _canonical_test_split(Path(config.data_dir), config)
    assert split is not None
    status_path = Path(config.output_dir) / "env" / "canonical_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text())
    assert payload["stems_sha256"] != ""
    assert payload["gt_content_sha256"] != ""
    assert payload["fallback_triggered"] is False
    assert payload["mirror_used"] == "huggingface"


def test_strict_mode_raises_when_both_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RRC fails AND HF returns None → DataError re-raised with strict-mode prefix."""
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})
    monkeypatch.setattr(sc_hf, "try_huggingface", _make_hf_returning_none())
    from data.sroie import _canonical_test_split
    with pytest.raises(DataError, match="strict mode"):
        _canonical_test_split(Path(config.data_dir), config)


def test_idempotent_short_circuit_requires_exact_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-populate test/img with 360 jpg → no short-circuit (stale-360 case)."""
    config = _min_config(tmp_path)
    img_dir = Path(config.data_dir) / "test" / "img"
    img_dir.mkdir(parents=True)
    for i in range(360):  # stale run wrote 360
        (img_dir / f"stale_{i:05d}.jpg").write_bytes(_JPEG_SOI)
    # HF fixture should be called — no short-circuit with 360 files
    call_log: list[str] = []

    def _recording_hf(workdir: Path, repo_id: str, revision: str) -> Path | None:
        call_log.append("called")
        return _make_hf_fixture()(workdir, repo_id, revision)

    _patch_downloads(monkeypatch, {})
    monkeypatch.setattr(sc_hf, "try_huggingface", _recording_hf)
    ensure_canonical_test_set(config, Path(config.data_dir))
    assert call_log == ["called"], "short-circuit fired with 360 files — gate is wrong"


def test_canonical_status_sidecar_records_huggingface(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """On HF success path, sidecar records mirror_used='huggingface'."""
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})
    monkeypatch.setattr(sc_hf, "try_huggingface", _make_hf_fixture())
    train_img = Path(config.data_dir) / "train" / "img"
    train_ent = Path(config.data_dir) / "train" / "entities"
    train_img.mkdir(parents=True)
    train_ent.mkdir(parents=True)
    for i in range(60):
        (train_img / f"T{i:05d}.jpg").write_bytes(_JPEG_SOI)
        (train_ent / f"T{i:05d}.txt").write_bytes(_GT_JSON)
    from data.sroie import _canonical_test_split
    split = _canonical_test_split(Path(config.data_dir), config)
    assert split is not None
    status_path = Path(config.output_dir) / "env" / "canonical_status.json"
    payload = json.loads(status_path.read_text())
    assert payload["mirror_used"] == "huggingface"
    assert payload["n_img_collected"] == _TASK3_TEST_COUNT
    assert payload["n_ent_collected"] == _TASK3_TEST_COUNT
    assert payload["fallback_triggered"] is False


def test_huggingface_hub_missing_returns_none_gracefully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If try_huggingface returns None (HF unavailable), propagates as DataError."""
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})
    monkeypatch.setattr(sc_hf, "try_huggingface", _make_hf_returning_none())
    with pytest.raises(DataError):
        ensure_canonical_test_set(config, Path(config.data_dir))


# --------------------------------------------------------------------------- #
# Sidecar tests (kept/updated)                                                 #
# --------------------------------------------------------------------------- #

def test_canonical_status_sidecar_written_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_canonical_test_split must persist canonical_status.json."""
    workdir = tmp_path / "src"
    workdir.mkdir()
    img_zip, gt_zip = _build_rrc_archives(workdir)
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {
        config.canonical_sroie_test_url: img_zip,
        config.canonical_sroie_gt_url: gt_zip,
    })
    train_img = Path(config.data_dir) / "train" / "img"
    train_ent = Path(config.data_dir) / "train" / "entities"
    train_img.mkdir(parents=True)
    train_ent.mkdir(parents=True)
    for i in range(80):
        (train_img / f"T{i:05d}.jpg").write_bytes(_JPEG_SOI)
        (train_ent / f"T{i:05d}.txt").write_bytes(_GT_JSON)
    from data.sroie import _canonical_test_split
    split = _canonical_test_split(Path(config.data_dir), config)
    assert split is not None
    status_path = Path(config.output_dir) / "env" / "canonical_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text())
    assert payload["fallback_triggered"] is False
    assert payload["mirror_used"] == "rrc"
    assert payload["n_img_collected"] == _TASK3_TEST_COUNT
    assert payload["n_ent_collected"] == _TASK3_TEST_COUNT
    assert payload["stems_sha256"] != ""
    assert payload["gt_content_sha256"] != ""


def test_canonical_status_sidecar_marks_fallback_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When canonical fetch fails, sidecar records fallback_triggered=True and re-raises."""
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})
    monkeypatch.setattr(sc_hf, "try_huggingface", _make_hf_returning_none())
    from data.sroie import _canonical_test_split
    with pytest.raises(DataError, match="strict mode"):
        _canonical_test_split(Path(config.data_dir), config)
    status_path = Path(config.output_dir) / "env" / "canonical_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text())
    assert payload["fallback_triggered"] is True
    assert payload["mirror_used"] == "none"
    assert "error" in payload


# --------------------------------------------------------------------------- #
# Paper stage tests (kept/updated)                                             #
# --------------------------------------------------------------------------- #

def test_paper_stage_refuses_advanced_caption_on_fallback(tmp_path: Path) -> None:
    """stages.paper must refuse 'advanced' when fallback_triggered=True."""
    from core.errors import EvalError
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="advanced")
    env_dir = Path(config.output_dir) / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "canonical_status.json").write_text(json.dumps({
        "mirror_used": "none", "n_img_collected": 0, "n_ent_collected": 0,
        "fallback_triggered": True, "error": "both mirrors failed",
    }))
    with pytest.raises(EvalError, match="fallback_triggered=True"):
        _assert_canonical_status_ok(config)


def test_paper_stage_refuses_advanced_caption_when_status_missing(
    tmp_path: Path,
) -> None:
    """Missing canonical_status.json with advanced+canonical is fatal."""
    from core.errors import EvalError
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="advanced")
    with pytest.raises(EvalError, match="is missing"):
        _assert_canonical_status_ok(config)


def test_paper_stage_accepts_rrc_mirror(tmp_path: Path) -> None:
    """fallback_triggered=False with mirror_used='rrc' → paper stage proceeds."""
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="advanced")
    env_dir = Path(config.output_dir) / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "canonical_status.json").write_text(json.dumps({
        "mirror_used": "rrc", "n_img_collected": 347, "n_ent_collected": 347,
        "fallback_triggered": False,
    }))
    _assert_canonical_status_ok(config)  # must not raise


def test_paper_stage_accepts_huggingface_mirror(tmp_path: Path) -> None:
    """fallback_triggered=False with mirror_used='huggingface' → paper stage proceeds."""
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="advanced")
    env_dir = Path(config.output_dir) / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "canonical_status.json").write_text(json.dumps({
        "mirror_used": "huggingface", "n_img_collected": 347, "n_ent_collected": 347,
        "fallback_triggered": False,
    }))
    _assert_canonical_status_ok(config)  # must not raise


def test_paper_stage_accepts_cached_mirror(tmp_path: Path) -> None:
    """fallback_triggered=False with mirror_used='cached' → paper stage proceeds."""
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="advanced")
    env_dir = Path(config.output_dir) / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "canonical_status.json").write_text(json.dumps({
        "mirror_used": "cached", "n_img_collected": 347, "n_ent_collected": 347,
        "fallback_triggered": False,
    }))
    _assert_canonical_status_ok(config)  # must not raise


def test_paper_stage_basic_variant_skips_check(tmp_path: Path) -> None:
    """paper_variant=basic is unaffected by canonical_status.json state."""
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="basic")
    _assert_canonical_status_ok(config)  # must not raise — no sidecar needed


# --------------------------------------------------------------------------- #
# Misc / structural tests                                                      #
# --------------------------------------------------------------------------- #

def test_ssl_context_uses_certifi_when_available() -> None:
    """`_ssl_context` returns an SSLContext with verify enabled."""
    import ssl as _ssl
    ctx = sc._ssl_context()
    assert isinstance(ctx, _ssl.SSLContext)
    assert ctx.verify_mode == _ssl.CERT_REQUIRED


def test_mirror_adapters_have_distinct_markers() -> None:
    """RRC and HF adapters use distinct image markers and entity globs."""
    assert isinstance(sc._RRC_ADAPTER, MirrorAdapter)
    assert isinstance(sc._HF_ADAPTER, MirrorAdapter)
    # Image path markers are distinct (key anti-contamination property)
    assert sc._RRC_ADAPTER.image_path_marker != sc._HF_ADAPTER.image_path_marker
    # Entity globs are distinct (*.txt vs *.json)
    assert sc._RRC_ADAPTER.entity_glob != sc._HF_ADAPTER.entity_glob
    assert sc._HF_ADAPTER.entity_format == "json"
    assert sc._RRC_ADAPTER.entity_format == "sroie_kv"


def test_canonical_status_dataclass_defaults() -> None:
    """CanonicalStatus.fallback_triggered defaults to False."""
    s = CanonicalStatus(mirror_used="rrc", n_img_collected=347, n_ent_collected=347)
    assert s.fallback_triggered is False
    assert s.stems_sha256 == ""
    assert s.gt_content_sha256 == ""


def test_pins_are_sentinels() -> None:
    """Merged PR ships in first-run mode — both pin constants are PIN_ON_FIRST_RUN."""
    from data.sroie_canonical import _RRC_TASK3_GT_SHA256, _RRC_TASK3_STEMS_SHA256
    assert _RRC_TASK3_STEMS_SHA256 == "PIN_ON_FIRST_RUN"
    assert _RRC_TASK3_GT_SHA256 == "PIN_ON_FIRST_RUN"


def test_hf_adapter_markers_match_materialiser_layout() -> None:
    """_HF_ADAPTER markers must exactly match what try_huggingface writes."""
    assert sc._HF_ADAPTER.image_path_marker == "img"
    assert sc._HF_ADAPTER.entity_path_marker == "entities"


def test_extract_fields_schema_variants() -> None:
    """_extract_fields resolves COMPANY/DATE/ADDRESS/TOTAL casing variants."""
    row: dict[str, object] = {
        "image_bytes": _JPEG_SOI,
        "COMPANY": "BAR", "DATE": "02/02/2025",
        "ADDRESS": "Y", "TOTAL": "2.00",
    }
    fields = sc_hf._extract_fields(row)
    assert fields == {"company": "BAR", "date": "02/02/2025", "address": "Y", "total": "2.00"}


def test_extract_stem_fallback() -> None:
    """_extract_stem falls back to ``f"X{idx:08d}"`` when no stem column.

    The X-prefix is an arbitrary deterministic convention chosen so the
    resulting filenames remain greppable in run artefacts; it does not
    encode RRC's actual receipt-id scheme.  Identity preservation is
    guaranteed by the GT content sha256 in the parent ensurer module
    (`data.sroie_canonical._RRC_TASK3_GT_SHA256`), not by stem format.
    """
    row: dict[str, object] = {"company": "FOO"}
    assert sc_hf._extract_stem(row, 5) == "X00000005"


def test_extract_stem_from_hf_image_path() -> None:
    """_extract_stem derives stem from HF image-struct ``path`` field."""
    row: dict[str, object] = {"image": {"path": "X51005200938.jpg"}}
    assert sc_hf._extract_stem(row, 0) == "X51005200938"


def test_extract_image_bytes_hf_struct() -> None:
    """_extract_image_bytes handles HF {"bytes": ..., "path": ...} struct."""
    row: dict[str, object] = {"image": {"bytes": _JPEG_SOI, "path": "x.jpg"}}
    assert sc_hf._extract_image_bytes(row) == _JPEG_SOI


def test_extract_gt_donut_style_string() -> None:
    """_extract_gt parses Donut-style ``ground_truth`` JSON string with gt_parse envelope."""
    row: dict[str, object] = {
        "ground_truth": json.dumps({
            "gt_parse": {
                "company": "ACME", "date": "01/01/2020",
                "address": "1 Main St", "total": "9.99",
            },
        }),
    }
    out = sc_hf._extract_gt(row)
    assert out == {
        "company": "ACME", "date": "01/01/2020",
        "address": "1 Main St", "total": "9.99",
    }


def test_extract_gt_gt_parses_list_envelope() -> None:
    """_extract_gt unwraps {"gt_parses": [{...}]} list envelope (multi-annotator)."""
    row: dict[str, object] = {
        "ground_truth": json.dumps({
            "gt_parses": [
                {"company": "A", "date": "d", "address": "x", "total": "1"},
                {"company": "B", "date": "d2", "address": "y", "total": "2"},
            ],
        }),
    }
    out = sc_hf._extract_gt(row)
    assert out is not None and out["company"] == "A"


def test_extract_gt_dict_input() -> None:
    """_extract_gt accepts a pre-parsed dict (HF JSON-typed columns)."""
    row: dict[str, object] = {
        "gt_parse": {"COMPANY": "X", "DATE": "d", "ADDRESS": "a", "TOTAL": "t"},
    }
    out = sc_hf._extract_gt(row)
    assert out == {"company": "X", "date": "d", "address": "a", "total": "t"}


def test_extract_gt_returns_none_for_missing_field() -> None:
    """_extract_gt returns None when a required field is absent."""
    row: dict[str, object] = {
        "ground_truth": json.dumps({"company": "A", "date": "d"}),  # no address/total
    }
    assert sc_hf._extract_gt(row) is None


def test_extract_gt_returns_none_for_invalid_json() -> None:
    """_extract_gt returns None on malformed JSON (per-row skip + count)."""
    row: dict[str, object] = {"ground_truth": "this is not json {{{"}
    assert sc_hf._extract_gt(row) is None


def test_extract_gt_flat_column_fallback() -> None:
    """_extract_gt falls back to flat-column extraction (legacy schema)."""
    row: dict[str, object] = {
        "company": "FOO", "date": "01/01/2024",
        "address": "X", "total": "1.00",
    }
    out = sc_hf._extract_gt(row)
    assert out == {
        "company": "FOO", "date": "01/01/2024",
        "address": "X", "total": "1.00",
    }


def test_hf_donut_style_ground_truth_parses_correctly(tmp_path: Path) -> None:
    """End-to-end: 347 Donut-style rows → 347/347 jpg+json with all four fields.

    Exercises the real :func:`_materialize_rows` (not the stub) by writing a
    parquet shard that mimics ``Metric-AI/icdar_sroie``'s schema (image bytes,
    ``ground_truth`` as a JSON string with the ``gt_parse`` envelope) and
    asserting the post-parse outputs satisfy D6's acceptance criterion.
    """
    pa = pytest.importorskip("pyarrow")
    pa_pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("datasets")
    snap_dir = tmp_path / "snap"
    data_dir = snap_dir / "data"
    data_dir.mkdir(parents=True)
    rows_company: list[str] = []
    rows_image: list[dict[str, object]] = []
    rows_gt: list[str] = []
    for i in range(_TASK3_TEST_COUNT):
        company = f"COMPANY_{i:03d}"
        rows_company.append(company)
        rows_image.append({"bytes": _JPEG_SOI, "path": f"X{i:08d}.jpg"})
        rows_gt.append(json.dumps({
            "gt_parse": {
                "company": company, "date": "01/01/2024",
                "address": f"{i} Main St", "total": f"{i}.99",
            },
        }))
    table = pa.table({
        "image": rows_image,
        "question": ["What is the total?"] * _TASK3_TEST_COUNT,
        "ground_truth": rows_gt,
    })
    pa_pq.write_table(table, data_dir / "test-00000-of-00001.parquet")
    img_dir = tmp_path / "img"
    ent_dir = tmp_path / "entities"
    img_dir.mkdir()
    ent_dir.mkdir()
    n = sc_hf._materialize_rows(snap_dir, img_dir, ent_dir)
    assert n == _TASK3_TEST_COUNT
    assert len(list(img_dir.glob("*.jpg"))) == _TASK3_TEST_COUNT
    json_files = sorted(ent_dir.glob("*.json"))
    assert len(json_files) == _TASK3_TEST_COUNT
    # Spot-check one entity file: all four canonical fields, non-empty strings.
    sample = json.loads(json_files[42].read_text())
    assert set(sample) == {"company", "date", "address", "total"}
    for k, v in sample.items():
        assert isinstance(v, str) and v, f"empty field {k}"


def test_hf_materialize_raises_on_unparseable_ground_truth(tmp_path: Path) -> None:
    """_materialize_rows raises DataError with sample bad rows when count < 347."""
    pa = pytest.importorskip("pyarrow")
    pa_pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("datasets")
    snap_dir = tmp_path / "snap"
    data_dir = snap_dir / "data"
    data_dir.mkdir(parents=True)
    table = pa.table({
        "image": [{"bytes": _JPEG_SOI, "path": "x.jpg"}] * _TASK3_TEST_COUNT,
        "ground_truth": ["not-valid-json {{{"] * _TASK3_TEST_COUNT,
    })
    pa_pq.write_table(table, data_dir / "test-00000-of-00001.parquet")
    img_dir = tmp_path / "img"
    ent_dir = tmp_path / "entities"
    img_dir.mkdir()
    ent_dir.mkdir()
    with pytest.raises(DataError, match="unparseable"):
        sc_hf._materialize_rows(snap_dir, img_dir, ent_dir)


def test_select_canonical_split_picks_347_row_split(tmp_path: Path) -> None:
    """_select_canonical_split prefers the split with exactly 347 rows."""
    pa = pytest.importorskip("pyarrow")
    pa_pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("datasets")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # train has 500 rows (wrong), test has 347 (correct), validation has 63.
    pa_pq.write_table(
        pa.table({"ground_truth": ["{}"] * 500}),
        data_dir / "train-00000-of-00001.parquet",
    )
    pa_pq.write_table(
        pa.table({"ground_truth": ["{}"] * _TASK3_TEST_COUNT}),
        data_dir / "test-00000-of-00001.parquet",
    )
    pa_pq.write_table(
        pa.table({"ground_truth": ["{}"] * 63}),
        data_dir / "validation-00000-of-00001.parquet",
    )
    ds = sc_hf._select_canonical_split(tmp_path)
    assert len(ds) == _TASK3_TEST_COUNT


def test_select_canonical_split_raises_when_no_split_matches(tmp_path: Path) -> None:
    """_select_canonical_split raises DataError when no split has 347 rows."""
    pa = pytest.importorskip("pyarrow")
    pa_pq = pytest.importorskip("pyarrow.parquet")
    pytest.importorskip("datasets")
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    pa_pq.write_table(
        pa.table({"ground_truth": ["{}"] * 100}),
        data_dir / "train-00000-of-00001.parquet",
    )
    with pytest.raises(DataError, match="no split with exactly 347 rows"):
        sc_hf._select_canonical_split(tmp_path)


def test_stems_sha256_deterministic(tmp_path: Path) -> None:
    """_stems_sha256 is deterministic and order-independent."""
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    for stem in ["C", "A", "B"]:
        (img_dir / f"{stem}.jpg").write_bytes(b"x")
    h1 = sc._stems_sha256(img_dir)
    h2 = hashlib.sha256(b"A\nB\nC").hexdigest()
    assert h1 == h2


def test_gt_content_sha256_handles_json_files(tmp_path: Path) -> None:
    """_gt_content_sha256 correctly digests .json entity files."""
    ent_dir = tmp_path / "entities"
    ent_dir.mkdir()
    (ent_dir / "X00001.json").write_text(
        '{"company": "A", "date": "d", "address": "a", "total": "t"}',
        encoding="utf-8",
    )
    h = sc._gt_content_sha256(ent_dir)
    assert len(h) == 64  # sha256 hex digest length


def test_gt_content_sha256_handles_txt_files(tmp_path: Path) -> None:
    """_gt_content_sha256 correctly digests .txt (RRC sroie_kv) entity files."""
    ent_dir = tmp_path / "entities"
    ent_dir.mkdir()
    (ent_dir / "X00001.txt").write_text(
        '{"company": "A", "date": "d", "address": "a", "total": "t"}',
        encoding="utf-8",
    )
    h_txt = sc._gt_content_sha256(ent_dir)
    # Should match the json variant (same content, different file extension)
    ent_dir2 = tmp_path / "entities2"
    ent_dir2.mkdir()
    (ent_dir2 / "X00001.json").write_text(
        '{"company": "A", "date": "d", "address": "a", "total": "t"}',
        encoding="utf-8",
    )
    h_json = sc._gt_content_sha256(ent_dir2)
    assert h_txt == h_json
