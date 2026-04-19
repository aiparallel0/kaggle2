"""Run ``models.pipeline_eval.eval_pipeline`` on the full test split and
print the same F1 numbers as ``main.py --stage eval`` — without running
the DONUT stage.

Point: ``scripts/diagnose_pipeline.py`` and ``eval_pipeline`` appear to
disagree on aggregate F1 (0.52 on the first 20 vs 0.11 on all 63).
Either the two code paths diverge, or something non-deterministic is
happening between runs. This script calls the *same* ``eval_pipeline``
function ``main.py`` uses, so its output is the source of truth for
what ``main.py --stage eval`` would print, minus the ~23-minute DONUT
stage.

Usage on vast.ai:

    cd /workspace/kaggle2
    python scripts/parity_eval_pipeline.py

If this prints ~0.11 → ``eval_pipeline`` is genuinely producing the bad
numbers and diagnose is somehow luckier.  If it prints ~0.5 →
``combined_metrics.json`` was written by an earlier bad eval pass and
the current checkpoints actually do well.  Either way we will have
eliminated one variable.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from core.config import load_config  # noqa: E402
from core.types import PipelinePaths  # noqa: E402
from data.sroie import download_sroie, load_or_create_split  # noqa: E402
from models.pipeline_eval import eval_pipeline  # noqa: E402


def main() -> None:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("parity")
    config = load_config("config.json")
    data_path = download_sroie(config)
    split_cache = Path(config.output_dir) / "split.json"
    data = load_or_create_split(data_path, config.seed, split_cache)
    paths = PipelinePaths(
        yolo=os.path.join(config.output_dir, "yolo", "run", "weights", "best.pt"),
        trocr=os.path.join(config.output_dir, "trocr"),
        assigner=os.path.join(config.output_dir, "assigner.pt"),
    )
    log.info("Running eval_pipeline on %d test receipts...", len(data.test))
    pm = eval_pipeline(paths, data.test, config)
    report = {
        "assigner_global_f1": pm.assigner.global_f1,
        "assigner_per_field_f1": pm.assigner.per_field_f1,
        "rulebased_global_f1": pm.rulebased.global_f1,
        "rulebased_per_field_f1": pm.rulebased.per_field_f1,
    }
    log.info("RESULT (via eval_pipeline — same function main.py calls):")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
