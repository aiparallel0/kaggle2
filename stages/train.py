"""Training stage: DONUT (optional), YOLO, TrOCR, AttentionAssigner.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: orchestrates the four training sub-stages in dependency order,
    wraps each one with telemetry, and validates the dataset split
    before any GPU work begins.  Honours ``config.skip_donut`` for
    pipeline-only "Phase 1" runs on constrained vast.ai instances.
"""
from __future__ import annotations

import logging

from core.errors import TrainError
from core.stage_telemetry import start_telem, stop_telem
from core.types import AssignerData, ExpConfig
from data.sroie import (
    download_sroie,
    extract_crops,
    extract_receipt_regions,
    load_or_create_split,
)
from models.donut_train import train_donut
from models.focus_train import train_assigner
from models.trocr_train import train_trocr
from models.yolo_train import train_yolo
from stages.common import write_pipeline_meta

log = logging.getLogger("kaggle2")


def _train_donut_stage(config: ExpConfig, data: object) -> None:
    """Execute the DONUT sub-stage unless ``skip_donut`` is set.

    Enforces the invariant that knowledge-distillation weights must be
    zero when the teacher is absent — otherwise the pipeline stages
    would attempt to distil from a non-existent ``results/donut``.
    """
    if config.skip_donut:
        log.info("skip_donut=True — Phase 1 mode: DONUT training suppressed. "
                 "KD losses will be disabled downstream (kd_*_weight must be 0).")
        if config.kd_attn_weight != 0.0 or config.kd_logits_weight != 0.0:
            raise TrainError(
                "skip_donut=True but kd_attn_weight or kd_logits_weight != 0. "
                "Phase 1 cannot distil from a missing teacher; set both to 0.",
            )
        return
    th, ev, t0 = start_telem(config, "donut")
    try:
        donut_path = train_donut(config, data)  # type: ignore[arg-type]
    finally:
        stop_telem(th, ev, t0, config, "donut")
    log.info("DONUT → %s", donut_path)


def stage_train(config: ExpConfig) -> None:
    """Run DONUT → YOLO → TrOCR → AttentionAssigner end-to-end.

    Each sub-stage writes its own checkpoint directory under
    ``config.output_dir`` and emits a telemetry JSONL + a cost summary
    (see :mod:`core.telem_stage`), keeping the paper's GPU-efficiency
    story reproducible without re-running the training itself.
    """
    log.info("=== Stage: train ===")
    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("Split: %d train / %d val / %d test",
             len(data.train), len(data.val), len(data.test))
    _train_donut_stage(config, data)
    th_y, ev_y, t0_y = start_telem(config, "yolo")
    try:
        yolo_path = train_yolo(config, data)
    finally:
        stop_telem(th_y, ev_y, t0_y, config, "yolo")
    log.info("YOLO  → %s", yolo_path)
    crops = extract_crops(data.train, config.fields)
    regions = extract_receipt_regions(data.train, config.fields)
    log.info("Extracted %d labeled crops / %d receipt region-groups",
             len(crops), len(regions))
    if not crops:
        raise TrainError("No labeled SROIE crops — check box/ annotations.")
    th_t, ev_t, t0_t = start_telem(config, "trocr")
    try:
        trocr_path = train_trocr(config, crops)
    finally:
        stop_telem(th_t, ev_t, t0_t, config, "trocr")
    log.info("TrOCR → %s", trocr_path)
    assigner_data = AssignerData(trocr_path=trocr_path, crops=crops, regions=regions)
    assigner_path = train_assigner(config, assigner_data)
    log.info("Assigner → %s", assigner_path)
    write_pipeline_meta(config)
