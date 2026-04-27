"""Per-mirror integration tests for canonical-SROIE MirrorAdapters.

Builds an in-memory fixture archive that mimics each known mirror's
on-disk layout (RRC dual-zip, docTR single-zip, plus a poisoned
variant carrying Task-1 box files + extra thumbnails) and asserts
that the placed-file counts come out at exactly 347/347 — the bar
that ``>= 347`` was hiding when 360 jpg + 0 txt landed on disk.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest

import data.sroie_canonical as sc
from core.config import load_config
from core.errors import DataError
from data.sroie_canonical import (
    _DOCTR_MIRROR_SHA256,
    _TASK3_TEST_COUNT,
    CanonicalStatus,
    MirrorAdapter,
    ensure_canonical_test_set,
)

# --------------------------------------------------------------------------- #
# Fixture archive builders                                                     #
# --------------------------------------------------------------------------- #

_JPEG_SOI = b"\xff\xd8\xff\xe0fakejpeg"
_GT_JSON = b'{"company": "FOO", "date": "01/01/2024", "address": "X", "total": "1.00"}'


def _build_rrc_archives(workdir: Path) -> tuple[Path, Path]:
    """Mimic the RRC dual-zip layout: images + GT in two separate archives."""
    img_zip = workdir / "rrc_img.zip"
    gt_zip = workdir / "rrc_gt.zip"
    with zipfile.ZipFile(img_zip, "w") as zf:
        for i in range(_TASK3_TEST_COUNT):
            zf.writestr(f"task3-test(347p)/X{i:05d}.jpg", _JPEG_SOI)
    with zipfile.ZipFile(gt_zip, "w") as zf:
        for i in range(_TASK3_TEST_COUNT):
            zf.writestr(f"test/entities/X{i:05d}.txt", _GT_JSON)
    return img_zip, gt_zip


def _build_doctr_archive(workdir: Path) -> Path:
    """Mimic the docTR single-zip layout: sroie2019_test/{images,annotations}/."""
    z = workdir / "doctr.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for i in range(_TASK3_TEST_COUNT):
            zf.writestr(f"sroie2019_test/images/X{i:05d}.jpg", _JPEG_SOI)
            zf.writestr(f"sroie2019_test/annotations/X{i:05d}.json", _GT_JSON)
    return z


def _build_doctr_archive_poisoned(workdir: Path) -> Path:
    """docTR layout PLUS 13 extra thumbnails + Task-1 box files (Bug-14 image side).

    Verifies that ``image_path_marker='images'`` rejects thumbnails and
    that the entity content-gate rejects Task-1 box .txt files.
    """
    z = workdir / "doctr_poisoned.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for i in range(_TASK3_TEST_COUNT):
            zf.writestr(f"sroie2019_test/images/X{i:05d}.jpg", _JPEG_SOI)
            zf.writestr(f"sroie2019_test/annotations/X{i:05d}.json", _GT_JSON)
        # 13 extra thumbnails outside images/ — must NOT be collected.
        for i in range(13):
            zf.writestr(f"sroie2019_test/thumbnails/T{i:03d}.jpg", _JPEG_SOI)
        # Task-1 box files inside annotations/ — must be content-rejected
        # (first non-whitespace byte is a digit, not '{').
        for i in range(5):
            zf.writestr(
                f"sroie2019_test/annotations/BOX{i:03d}.json",
                b"10,20,300,20,300,60,10,60,COMPANY:FOO\n",
            )
    return z


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
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
        "canonical_sroie_mirror_url": "https://test.invalid/mirror.zip",
    }
    cfg.update(overrides)
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    os.environ.pop("KAGGLE2_PAPER_VARIANT", None)
    return load_config(str(p))


def _patch_downloads(
    monkeypatch: pytest.MonkeyPatch, fakes: dict[str, Path],
    sha_override: str | None = None,
) -> None:
    """Make ``_download`` copy from ``fakes[url]`` instead of hitting network."""
    import shutil

    def fake_download(url: str, dst: Path, timeout: float = 60.0) -> None:
        if url not in fakes:
            from core.errors import DataError as _DE
            raise _DE(f"fixture: no fake for {url}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(fakes[url], dst)

    monkeypatch.setattr(sc, "_download", fake_download)
    if sha_override is not None:
        monkeypatch.setattr(sc, "_DOCTR_MIRROR_SHA256", sha_override)


# --------------------------------------------------------------------------- #
# Tests                                                                        #
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


def test_doctr_adapter_collects_exactly_347(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """docTR single-zip layout → 347 img + 347 .json ent (NOT 360 / 0)."""
    workdir = tmp_path / "src"
    workdir.mkdir()
    z = _build_doctr_archive(workdir)
    config = _min_config(tmp_path)
    sha = sc._sha256_file(z)
    # Force RRC primary to fail so the docTR adapter is exercised.
    _patch_downloads(monkeypatch, {
        config.canonical_sroie_mirror_url: z,
    }, sha_override=sha)
    data_path = Path(config.data_dir)
    status = ensure_canonical_test_set(config, data_path)
    assert status.mirror_used == "doctr"
    assert status.n_img_collected == _TASK3_TEST_COUNT
    assert status.n_ent_collected == _TASK3_TEST_COUNT
    img_count = len(list((data_path / "test" / "img").glob("*.jpg")))
    ent_count = len(list((data_path / "test" / "entities").glob("*.json")))
    assert img_count == _TASK3_TEST_COUNT
    assert ent_count == _TASK3_TEST_COUNT


def test_doctr_adapter_rejects_poison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thumbnails + Task-1 box files in the docTR archive must be filtered out."""
    workdir = tmp_path / "src"
    workdir.mkdir()
    z = _build_doctr_archive_poisoned(workdir)
    config = _min_config(tmp_path)
    sha = sc._sha256_file(z)
    _patch_downloads(monkeypatch, {
        config.canonical_sroie_mirror_url: z,
    }, sha_override=sha)
    data_path = Path(config.data_dir)
    status = ensure_canonical_test_set(config, data_path)
    # If the marker / content gates were lax, we'd see 360 img and 352 ent.
    assert status.n_img_collected == _TASK3_TEST_COUNT, (
        "image_path_marker failed to scope out thumbnails")
    assert status.n_ent_collected == _TASK3_TEST_COUNT, (
        "entity content-gate failed to reject Task-1 box .json files")


def test_strict_canonical_raises_on_total_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When BOTH mirrors are unreachable AND canonical is requested, raise."""
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})  # no fakes → every download fails
    with pytest.raises(DataError, match="both primary"):
        ensure_canonical_test_set(config, Path(config.data_dir))


def test_canonical_status_sidecar_written_on_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``data.sroie._canonical_test_split`` must persist canonical_status.json."""
    workdir = tmp_path / "src"
    workdir.mkdir()
    img_zip, gt_zip = _build_rrc_archives(workdir)
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {
        config.canonical_sroie_test_url: img_zip,
        config.canonical_sroie_gt_url: gt_zip,
    })
    # Pre-populate train/ so the canonical-test code path has training
    # receipts to pair with the canonical test set.
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


def test_canonical_status_sidecar_marks_fallback_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When canonical fetch fails, sidecar records fallback_triggered=True
    and the strict path re-raises (no silent split downgrade).
    """
    config = _min_config(tmp_path)
    _patch_downloads(monkeypatch, {})
    from data.sroie import _canonical_test_split
    with pytest.raises(DataError, match="strict mode"):
        _canonical_test_split(Path(config.data_dir), config)
    status_path = Path(config.output_dir) / "env" / "canonical_status.json"
    assert status_path.exists()
    payload = json.loads(status_path.read_text())
    assert payload["fallback_triggered"] is True
    assert payload["mirror_used"] == "none"
    assert "error" in payload


def test_paper_stage_refuses_advanced_caption_on_fallback(tmp_path: Path) -> None:
    """stages.paper must refuse to run with paper_variant=advanced when
    canonical_status.json reports fallback_triggered=True.
    """
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
    """Missing canonical_status.json with advanced+canonical is also fatal."""
    from core.errors import EvalError
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="advanced")
    with pytest.raises(EvalError, match="is missing"):
        _assert_canonical_status_ok(config)


def test_paper_stage_accepts_advanced_caption_on_success(tmp_path: Path) -> None:
    """fallback_triggered=False in the sidecar → paper stage proceeds."""
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="advanced")
    env_dir = Path(config.output_dir) / "env"
    env_dir.mkdir(parents=True)
    (env_dir / "canonical_status.json").write_text(json.dumps({
        "mirror_used": "rrc", "n_img_collected": 347, "n_ent_collected": 347,
        "fallback_triggered": False,
    }))
    _assert_canonical_status_ok(config)  # must not raise


def test_paper_stage_basic_variant_skips_check(tmp_path: Path) -> None:
    """paper_variant=basic is unaffected by canonical_status.json state."""
    from stages.paper import _assert_canonical_status_ok
    config = _min_config(tmp_path, paper_variant="basic")
    _assert_canonical_status_ok(config)  # must not raise — no sidecar needed


def test_ssl_context_uses_certifi_when_available() -> None:
    """`_ssl_context` returns an SSLContext with verify enabled."""
    import ssl as _ssl
    ctx = sc._ssl_context()
    assert isinstance(ctx, _ssl.SSLContext)
    assert ctx.verify_mode == _ssl.CERT_REQUIRED


def test_mirror_adapters_have_distinct_markers() -> None:
    """Per-mirror markers prevent cross-mirror contamination by construction."""
    assert isinstance(sc._RRC_ADAPTER, MirrorAdapter)
    assert isinstance(sc._DOCTR_ADAPTER, MirrorAdapter)
    assert sc._RRC_ADAPTER.image_path_marker != sc._DOCTR_ADAPTER.image_path_marker
    assert sc._RRC_ADAPTER.entity_path_marker != sc._DOCTR_ADAPTER.entity_path_marker
    assert sc._DOCTR_ADAPTER.entity_format == "json"
    assert sc._RRC_ADAPTER.entity_format == "sroie_kv"


def test_canonical_status_dataclass_defaults() -> None:
    """CanonicalStatus.fallback_triggered defaults to False."""
    s = CanonicalStatus(mirror_used="rrc", n_img_collected=347, n_ent_collected=347)
    assert s.fallback_triggered is False


def test_doctr_sha_pinned() -> None:
    """The docTR mirror sha256 stays pinned (catches accidental rotation)."""
    assert _DOCTR_MIRROR_SHA256 == (
        "41b3c746a20226fddc80d86d4b2a903d43b5be4f521dd1bbe759dbf8844745e2"
    )
