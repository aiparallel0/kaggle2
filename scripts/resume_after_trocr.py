"""Disk-full recovery for a train run that crashed at trainer.save_model.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: idempotent recovery script that frees disk, promotes the best TrOCR
    checkpoint, and runs the remaining train_assigner step after a
    vast.ai ENOSPC crash.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.seed import seed_everything  # noqa: E402
from core.types import AssignerData, ExpConfig  # noqa: E402
from data.sroie import (  # noqa: E402
    download_sroie,
    extract_crops,
    extract_receipt_regions,
    load_or_create_split,
)
from models.focus_train import train_assigner  # noqa: E402
from scripts.resume_cleanup import cleanup_disk, print_disk  # noqa: E402
from scripts.resume_promote import promote_trocr  # noqa: E402

log = logging.getLogger("resume")


def _remaining_train_steps(config: ExpConfig) -> None:
    """Run the train-stage steps that crashed short of: assigner + meta."""
    data_path = download_sroie(config)
    split_cache = Path(config.output_dir) / "split.json"
    data = load_or_create_split(config, data_path)
    log.info("Split cache %s → %d train / %d val / %d test",
             split_cache, len(data.train), len(data.val), len(data.test))
    crops = extract_crops(data.train, config.fields)
    regions = extract_receipt_regions(data.train, config.fields)
    if not crops:
        raise SystemExit(
            "No labeled SROIE crops after resume — check data/ cache integrity.",
        )
    log.info("%d crops / %d region-groups", len(crops), len(regions))

    assigner_marker = Path(config.output_dir) / "assigner.pt"
    if assigner_marker.exists():
        log.info("%s already exists — skipping train_assigner.", assigner_marker)
    else:
        trocr_path = str(Path(config.output_dir) / "trocr")
        out = train_assigner(
            config,
            AssignerData(trocr_path=trocr_path, crops=crops, regions=regions),
        )
        log.info("Assigner → %s", out)

    meta_path = Path(config.output_dir) / "pipeline_meta.json"
    meta_path.write_text(json.dumps({"yolo_image_size": config.yolo_image_size}))
    log.info("pipeline_meta.json → %s", meta_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument(
        "--skip-cleanup", action="store_true",
        help="Skip disk cleanup (DONUT checkpoints, yolo_data, caches).",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)
    seed_everything(config.seed)

    print_disk("before cleanup:")
    if not args.skip_cleanup:
        cleanup_disk(config)
    promote_trocr(config)
    print_disk("after promotion:")
    _remaining_train_steps(config)
    print_disk("after resume:")


if __name__ == "__main__":
    main()
