"""One-shot DONUT-eval harness that always terminates and writes diagnostics.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: loads the DONUT checkpoint from the latest run directory (or a
    specified one), runs :func:`eval_donut`, writes
    ``donut_eval_diag.json`` with full per-receipt diagnostics, and
    prints DONUT F1 — **without** calling :func:`validate_f1`, so the
    script always terminates and dumps artifacts even when F1 < 0.50.

This script is the recommended first step after a sub-floor DONUT run:
it preserves ``donut_eval_diag.json`` for forensic inspection of which
Bug (1/2/3/8) is responsible, and reports the exact start/eos token ids,
lm_head vocab size, and sample-level parsed vs. GT diffs.

Usage (copy-paste ready):

    # target the most recently modified run under runs/
    python -m scripts.donut_eval_only

    # target a specific run directory explicitly
    python -m scripts.donut_eval_only config.json --run-dir runs/20260426T174720Z-7c2e37c

    # inspect diagnostics
    jq '.lm_head_out_features, .tokenizer_vocab_size, .decoder_start_token_id' \\
        runs/<run_id>/donut/donut_eval_diag.json
    jq '.samples[0]' runs/<run_id>/donut/donut_eval_diag.json

To re-run the full pipeline with the F1 guard downgraded to a warning
(so combined_metrics.json is still written), set the env var and run
the eval stage:

    KAGGLE2_F1_WARN_ONLY=1 python main.py --stage eval
"""
from __future__ import annotations

import logging
import os
import sys

from core.config import load_config
from core.runlayout import latest_run
from data.sroie import download_sroie, load_or_create_split
from models.donut_eval import eval_donut

log = logging.getLogger("kaggle2")


def main(argv: list[str] | None = None) -> int:
    """Run DONUT eval on the latest (or specified) run dir; write diag artifact."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = list(argv if argv is not None else sys.argv[1:])
    config_path = args[0] if args and not args[0].startswith("-") else "configs/default.json"

    # Parse --run-dir / -r <path>
    run_dir: str | None = None
    for i, a in enumerate(args):
        if a in ("--run-dir", "-r") and i + 1 < len(args):
            run_dir = args[i + 1]
            break

    config = load_config(config_path)

    # Resolve run directory: --run-dir > KAGGLE2_RUN_DIR env > latest under runs/
    if run_dir is None:
        run_dir = os.environ.get("KAGGLE2_RUN_DIR")
    if run_dir is None:
        latest = latest_run("runs")
        if latest is None:
            log.error(
                "No run directories found under runs/. "
                "Pass --run-dir <path> or set KAGGLE2_RUN_DIR.",
            )
            return 1
        run_dir = str(latest)

    log.info("donut_eval_only: targeting run dir %s", run_dir)
    config.output_dir = run_dir

    data_path = download_sroie(config)
    data = load_or_create_split(config, data_path)
    log.info("donut_eval_only: %d test receipts", len(data.test))

    # eval_donut itself does not call validate_f1 — this script intentionally
    # skips validate_f1 so it always terminates and writes donut_eval_diag.json.
    metrics, _, _ = eval_donut(config, data.test)

    log.info(
        "DONUT  F1=%.4f  NED=%.4f  EM=%.4f",
        metrics.global_f1, metrics.global_ned, metrics.global_em,
    )
    diag_path = os.path.join(run_dir, "donut", "donut_eval_diag.json")
    log.info("Diagnostics → %s", diag_path)
    log.info(
        "donut_eval_only: validate_f1 guard was skipped.  "
        "Run `python main.py --stage eval` (or set "
        "KAGGLE2_F1_WARN_ONLY=1 for a warn-only full pipeline) "
        "for the gated evaluation.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
