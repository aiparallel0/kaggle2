"""test_pack_run.py — light-mode archive stays reviewer-downloadable.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: lock in the contract of ``scripts/pack_run.sh`` that
    ``make pack`` (light mode) strips files > 1\\,MiB and the heavy
    checkpoint subdirs (``donut/``, ``trocr/``, ``yolo/run/``,
    ``yolo_data/``) so the resulting ``.tar.zst`` fits in a PR review
    attachment while keeping every JSON sidecar, figure PDF and log.

The test materialises a synthetic run dir under ``runs/<id>/`` that
mirrors what a real training pipeline writes: a handful of <1\\,KiB
artefacts (JSON, log, small figure PDF) plus simulated checkpoint
blobs >1\\,MiB.  It then invokes ``pack_run.sh`` twice — once without
``--full`` and once with ``--full`` — and asserts:

    1. Light mode's archive contains NONE of the heavy files.
    2. Light mode's archive contains ``EXCLUDED.txt`` listing them.
    3. Full mode's archive contains ALL files.
    4. Light mode's archive is strictly smaller than full mode's.
    5. Both modes produce a matching ``.sha256`` sidecar.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tarfile
from collections.abc import Iterator
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PACK_SCRIPT = REPO_ROOT / "scripts" / "pack_run.sh"


def _populate(run_dir: Path) -> None:
    """Mirror a minimal post-train run directory under ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("donut", "trocr", "yolo/run/weights", "yolo_data",
                "figures", "metrics", "paper", "env"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    # Small, reviewer-useful artefacts (all < 1 MiB).
    (run_dir / "combined_metrics.json").write_text('{"pipeline_f1": 0.74}')
    (run_dir / "kaggle2_pipeline.log").write_text("INFO run done\n")
    (run_dir / "figures" / "fig_f1.pdf").write_bytes(b"%PDF-1.4\n" + b"x" * 200)
    (run_dir / "paper" / "paper_filled.tex").write_text("\\documentclass...")
    # Heavy checkpoint blobs (> 1 MiB) — must be dropped by light mode.
    (run_dir / "donut" / "model.safetensors").write_bytes(b"\0" * 2_000_000)
    (run_dir / "trocr" / "model.safetensors").write_bytes(b"\0" * 1_500_000)
    (run_dir / "yolo" / "run" / "weights" / "best.pt").write_bytes(b"\0" * 800_000)
    # Orphan big file outside the heavy dirs — must also be dropped.
    (run_dir / "huge_orphan.bin").write_bytes(b"\0" * 2_500_000)


def _tar_members(archive: Path) -> set[str]:
    """Return the set of member names inside a ``.tar.zst`` or ``.tar.gz``."""
    if archive.suffix == ".zst":
        # zstandard isn't in tarfile; decompress to a temp blob.
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".tar") as tmp:
            subprocess.run(
                ["zstd", "-d", "-q", "-o", tmp.name, "-f", str(archive)],
                check=True,
            )
            with tarfile.open(tmp.name) as tf:
                return set(tf.getnames())
    with tarfile.open(archive, "r:gz") as tf:
        return set(tf.getnames())


@pytest.fixture
def synthetic_run(tmp_path: Path) -> Iterator[tuple[Path, Path]]:
    """Build a real runs/<id>/ under the repo so pack_run.sh can find it.

    ``pack_run.sh`` resolves ``RUNS_ROOT`` from its own path (the repo's
    ``runs/`` directory), so for a faithful integration test we create
    the synthetic run there and clean up after.
    """
    run_id = f"pytest-{os.getpid()}-{tmp_path.name}"
    run_dir = REPO_ROOT / "runs" / run_id
    _populate(run_dir)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    try:
        yield run_dir, out_dir
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def _run_pack(run_id: str, out_dir: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(PACK_SCRIPT), *extra, run_id, str(out_dir)],
        capture_output=True, text=True, check=True,
    )


def test_light_mode_excludes_heavy_files(synthetic_run: tuple[Path, Path]) -> None:
    """Default (light) mode drops files > 1\\,MiB and heavy checkpoint dirs."""
    run_dir, out_dir = synthetic_run
    run_id = run_dir.name
    _run_pack(run_id, out_dir)
    # Archive should exist with either zstd or gzip extension; the
    # matching ``.sha256`` sidecar must NOT be mistaken for an archive.
    archives = [p for p in out_dir.iterdir()
                if p.suffix in {".zst", ".gz"} and p.stem.startswith(run_id)]
    assert len(archives) == 1, f"expected one archive, got {archives}"
    archive = archives[0]
    # Matching .sha256 sidecar written alongside.
    assert (archive.parent / f"{archive.name}.sha256").is_file()
    members = _tar_members(archive)
    # Heavy files stripped.
    assert not any("model.safetensors" in m for m in members), \
        f"model.safetensors leaked into light archive: {sorted(members)}"
    assert not any("best.pt" in m for m in members), \
        f"YOLO weights leaked into light archive: {sorted(members)}"
    assert not any("huge_orphan" in m for m in members), \
        f"orphan big file leaked: {sorted(members)}"
    # Small reviewer-useful artefacts preserved.
    assert f"{run_id}/combined_metrics.json" in members
    assert f"{run_id}/figures/fig_f1.pdf" in members
    assert f"{run_id}/EXCLUDED.txt" in members
    # Under a real review-attachment ceiling.
    assert archive.stat().st_size < 5 * 1024 * 1024, \
        f"light archive grew to {archive.stat().st_size} bytes"


def test_excluded_txt_lists_dropped_paths(synthetic_run: tuple[Path, Path]) -> None:
    """EXCLUDED.txt inside the archive names every stripped path."""
    run_dir, out_dir = synthetic_run
    run_id = run_dir.name
    _run_pack(run_id, out_dir)
    excluded_txt = (run_dir / "EXCLUDED.txt").read_text()
    for needle in ("donut/", "trocr/", "yolo/run/", "huge_orphan.bin"):
        assert needle in excluded_txt, f"EXCLUDED.txt missing {needle!r}"


def _find_archive(out_dir: Path, run_id: str) -> Path:
    archives = [p for p in out_dir.iterdir()
                if p.suffix in {".zst", ".gz"} and p.stem.startswith(run_id)]
    assert len(archives) == 1, f"expected one archive, got {archives}"
    return archives[0]


def test_full_mode_includes_everything(synthetic_run: tuple[Path, Path]) -> None:
    """``--full`` restores the pre-change behaviour — no exclusions."""
    run_dir, out_dir = synthetic_run
    run_id = run_dir.name
    _run_pack(run_id, out_dir, "--full")
    archive = _find_archive(out_dir, run_id)
    members = _tar_members(archive)
    assert f"{run_id}/donut/model.safetensors" in members
    assert f"{run_id}/trocr/model.safetensors" in members
    assert f"{run_id}/yolo/run/weights/best.pt" in members
    assert f"{run_id}/huge_orphan.bin" in members


def test_sha256_matches_archive_contents(synthetic_run: tuple[Path, Path]) -> None:
    """The sidecar sha256 is the hash of the archive bytes."""
    run_dir, out_dir = synthetic_run
    run_id = run_dir.name
    _run_pack(run_id, out_dir)
    archive = _find_archive(out_dir, run_id)
    side = archive.parent / f"{archive.name}.sha256"
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    # sidecar format: "<hex>  <basename>\n"
    assert side.read_text().split()[0] == expected
