"""Decomposed training stages for parallel multi-seed sweeps.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: split the monolithic ``stages.train.stage_train`` into two
    independently-cacheable phases so a multi-seed sweep can amortise
    the heavy backbone training across seeds:

    * :func:`stage_train_backbone` runs DONUT + YOLO + TrOCR (the
      seed-insensitive backbones in the sense that for a fair
      multi-seed comparison of the AttentionAssigner contribution,
      one freezes the upstream OCR/detector pair).  Output:
      ``runs/<id>/{donut,yolo,trocr}/``.
    * :func:`stage_train_assigner_only` consumes a previously trained
      backbone (by symlink or rsync) and trains ONLY the
      AttentionAssigner.  This is the per-seed loop step: a 1–2-min
      job on H100 vs the 60–80-min full pipeline.

The original :func:`stages.train.stage_train` remains the canonical
single-machine entry point and is unchanged.  Local RTX-4090 users
who run ``make all`` see no behavioural difference.

This module is deliberately additive — every existing test, config,
and CLI flag continues to work.  The decomposition is opt-in via
``--stage train_backbone`` / ``--stage train_assigner`` on
``main.py``.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

from core.errors import TrainError
from core.stage_telemetry import start_telem, stop_telem
from core.types import AssignerData, DataSplit, ExpConfig
from data.sroie import (
    download_sroie,
    extract_crops,
    extract_receipt_regions,
    load_or_create_split,
)
from models.focus_train import train_assigner
from models.trocr_train import train_trocr
from models.yolo_train import train_yolo
from stages.common import write_pipeline_meta
from stages.train import _train_donut_stage

log = logging.getLogger("kaggle2")


def _backbone_complete(output_dir: str) -> dict[str, bool]:
    """Return which backbone components are already on disk."""
    out = Path(output_dir)
    return {
        "donut": (out / "donut" / "config.json").exists(),
        "yolo": (out / "yolo" / "run" / "weights" / "best.pt").exists(),
        "trocr": (out / "trocr" / "config.json").exists(),
    }


def stage_train_backbone(config: ExpConfig) -> None:
    """Train the seed-stable backbones (DONUT, YOLO, TrOCR) only.

    No AttentionAssigner.  Suitable for a per-dataset cache: one
    backbone run feeds N per-seed assigner runs downstream.  Honours
    ``config.skip_donut`` for pipeline-only deployments.

    The output directory mirrors :func:`stages.train.stage_train` so
    a downstream :func:`stage_train_assigner_only` call (or the
    legacy :func:`stages.train.stage_train`) can pick up the same
    artefacts without renaming.
    """
    log.info("=== Stage: train_backbone ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info(
        "Split: %d train / %d val / %d test",
        len(data.train), len(data.val), len(data.test),
    )
    _train_donut_stage(config, data)
    th_y, ev_y, t0_y = start_telem(config, "yolo")
    try:
        yolo_path = train_yolo(config, data)
    finally:
        stop_telem(th_y, ev_y, t0_y, config, "yolo")
    log.info("YOLO  → %s", yolo_path)
    crops = extract_crops(data.train, config.fields)
    if not crops:
        raise TrainError("No labeled SROIE crops — check box/ annotations.")
    th_t, ev_t, t0_t = start_telem(config, "trocr")
    try:
        trocr_path = train_trocr(config, crops)
    finally:
        stop_telem(th_t, ev_t, t0_t, config, "trocr")
    log.info("TrOCR → %s", trocr_path)
    # Persist a small manifest so downstream assigner-only runs can
    # validate the backbone is intact before they try to load it.
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "trocr_path": trocr_path,
        "yolo_path": yolo_path,
        "components": _backbone_complete(config.output_dir),
        "skip_donut": config.skip_donut,
        "epochs_donut": config.epochs_donut,
        "epochs_yolo": config.epochs_yolo,
        "epochs_trocr": config.epochs_trocr,
    }
    with open(out_dir / "backbone_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Backbone manifest → %s", out_dir / "backbone_manifest.json")


# --- Per-component stages (true parallel execution) -----------------------
# The legacy stage_train_backbone runs DONUT -> YOLO -> TrOCR sequentially
# inside one process, which prevents true multi-GPU parallelism (you can
# DDP DONUT across 4 GPUs but YOLO and TrOCR sit idle until DONUT finishes).
# These per-component entry points let an outer orchestrator
# (scripts/single_instance_swarm.sh) launch each component as its own
# torchrun / python process pinned to a disjoint GPU subset, so the three
# heavy backbones train concurrently and wall clock = max() not sum().


def _ensure_data(config: ExpConfig) -> DataSplit:
    """Idempotent dataset download + split; safe under concurrent calls
    because :func:`download_sroie` early-returns when the cache is
    populated.  Pre-warm with :func:`stage_prepare_data` once before
    fanning out parallel processes to avoid race-time clones.
    """
    data_path = download_sroie(config)
    return load_or_create_split(config, data_path)


def stage_prepare_data(config: ExpConfig) -> None:
    """Download SROIE + materialise the persistent split BEFORE the
    parallel-component phase, so concurrent processes don't race on
    ``git clone`` or ``split.json`` write.
    """
    log.info("=== Stage: prepare_data ===")
    data = _ensure_data(config)
    log.info(
        "Split: %d train / %d val / %d test",
        len(data.train), len(data.val), len(data.test),
    )


def stage_train_donut(config: ExpConfig) -> None:
    """Train DONUT only; honours ``config.skip_donut``."""
    log.info("=== Stage: train_donut ===")
    data = _ensure_data(config)
    _train_donut_stage(config, data)


def stage_train_yolo(config: ExpConfig) -> None:
    """Train YOLOv8 only."""
    log.info("=== Stage: train_yolo ===")
    data = _ensure_data(config)
    th, ev, t0 = start_telem(config, "yolo")
    try:
        yolo_path = train_yolo(config, data)
    finally:
        stop_telem(th, ev, t0, config, "yolo")
    log.info("YOLO  → %s", yolo_path)


def stage_train_trocr(config: ExpConfig) -> None:
    """Train TrOCR only.  Independent of YOLO/DONUT — uses GT box crops."""
    log.info("=== Stage: train_trocr ===")
    data = _ensure_data(config)
    crops = extract_crops(data.train, config.fields)
    if not crops:
        raise TrainError("No labeled SROIE crops — check box/ annotations.")
    th, ev, t0 = start_telem(config, "trocr")
    try:
        trocr_path = train_trocr(config, crops)
    finally:
        stop_telem(th, ev, t0, config, "trocr")
    log.info("TrOCR → %s", trocr_path)


def stage_write_backbone_manifest(config: ExpConfig) -> None:
    """Write ``backbone_manifest.json`` after the per-component fan-out
    converges.  Called by the orchestrator once train_donut + train_yolo +
    train_trocr have all finished and joined their output into the same
    run directory.
    """
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "trocr_path": str(out_dir / "trocr"),
        "yolo_path": str(out_dir / "yolo" / "run" / "weights" / "best.pt"),
        "components": _backbone_complete(config.output_dir),
        "skip_donut": config.skip_donut,
        "epochs_donut": config.epochs_donut,
        "epochs_yolo": config.epochs_yolo,
        "epochs_trocr": config.epochs_trocr,
    }
    with open(out_dir / "backbone_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    log.info("Backbone manifest → %s", out_dir / "backbone_manifest.json")


def _resolve_backbone_source(config: ExpConfig) -> str:
    """Resolve the backbone source dir from env or config.

    Search order:
      1. env ``KAGGLE2_BACKBONE_FROM`` — set by ``scripts/run_seed.sh``
      2. ``config.output_dir`` itself — when the same instance trained
         the backbone earlier (legacy single-machine flow)
    """
    src = os.environ.get("KAGGLE2_BACKBONE_FROM", "").strip()
    if src and Path(src).exists():
        return src
    return config.output_dir


def _link_or_copy_backbone(src: str, dst: str) -> None:
    """Materialise ``src``'s {donut,yolo,trocr}/ inside ``dst``.

    Symlink when the source filesystem allows it (vast.ai shared
    storage / local rsync), copy otherwise.  Idempotent — already-
    populated targets are left alone so a re-run of the assigner
    stage doesn't re-download a 200 MB DONUT.
    """
    src_p, dst_p = Path(src), Path(dst)
    dst_p.mkdir(parents=True, exist_ok=True)
    for name in ("donut", "yolo", "trocr"):
        s = src_p / name
        d = dst_p / name
        if not s.exists() or d.exists():
            continue
        try:
            os.symlink(s.resolve(), d)
            log.info("Backbone link %s → %s", d, s.resolve())
        except OSError:
            shutil.copytree(s, d)
            log.info("Backbone copy %s → %s", s, d)


def stage_train_assigner_only(config: ExpConfig) -> None:
    """Train ONLY the AttentionAssigner against a pre-existing backbone.

    Resolves the backbone path from ``KAGGLE2_BACKBONE_FROM`` env or
    falls back to ``config.output_dir`` (legacy in-place training).
    Validates the backbone manifest, materialises the components into
    the run directory (symlink first, copy on failure), then runs the
    same :func:`models.focus_train.train_assigner` the legacy stage
    invokes — guaranteeing bit-identical assigner training to a
    monolithic ``stage_train`` invocation on the same hardware.
    """
    log.info("=== Stage: train_assigner ===")
    src = _resolve_backbone_source(config)
    if not Path(src).exists():
        raise TrainError(
            f"Backbone source not found: {src}. "
            "Set KAGGLE2_BACKBONE_FROM=<path> or run --stage train_backbone first.",
        )
    _link_or_copy_backbone(src, config.output_dir)
    have = _backbone_complete(config.output_dir)
    required = {"yolo": True, "trocr": True}
    missing = [name for name in required if not have.get(name, False)]
    if missing:
        raise TrainError(
            f"Backbone incomplete in {config.output_dir}: missing {missing}. "
            "Did the train_backbone stage finish successfully?",
        )
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    crops = extract_crops(data.train, config.fields)
    regions = extract_receipt_regions(data.train, config.fields)
    if not crops:
        raise TrainError("No labeled SROIE crops — check box/ annotations.")
    trocr_path = str(Path(config.output_dir) / "trocr")
    assigner_data = AssignerData(
        trocr_path=trocr_path, crops=crops, regions=regions,
    )
    assigner_path = train_assigner(config, assigner_data)
    log.info("Assigner → %s", assigner_path)
    write_pipeline_meta(config)


__all__ = [
    "stage_prepare_data",
    "stage_train_assigner_only",
    "stage_train_backbone",
    "stage_train_donut",
    "stage_train_trocr",
    "stage_train_yolo",
    "stage_write_backbone_manifest",
]
