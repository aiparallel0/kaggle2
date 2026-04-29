"""Telemetry lifecycle helpers used by the top-level training stages.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: wraps ``core.telemetry.start_sampler`` / ``stop_sampler`` with
    best-effort error handling so that a missing ``nvidia-smi``, a
    permission error, or any unexpected ``Exception`` never aborts the
    training stage it instruments.  Also writes ``cost_<stage>.json``
    summaries priced at ``$VASTAI_RATE_USD_HR`` (default $0.50/hr).
"""
from __future__ import annotations

import json
import logging
import os
import time
from threading import Event, Thread

from core.cost import summarise as cost_summarise
from core.telemetry import start_sampler, stop_sampler
from core.types import ExpConfig

log = logging.getLogger("kaggle2")

# vast.ai default hourly rate; override via the ``VASTAI_RATE_USD_HR``
# environment variable if the rented instance runs at a different price.
_VASTAI_RATE_USD_HR: float = 0.50


def start_telem(
    config: ExpConfig, stage: str,
) -> tuple[Thread | None, Event | None, float]:
    """Start a telemetry sampler for *stage*; return (thread, event, t0).

    Any exception is caught and logged as a warning so that a missing
    ``nvidia-smi`` or permission error never aborts training.
    """
    out_path = os.path.join(config.output_dir, f"telemetry_{stage}.jsonl")
    try:
        thread, event = start_sampler(out_path, interval_s=5.0)
        return thread, event, time.monotonic()
    except Exception as exc:  # noqa: BLE001
        log.warning("Telemetry start failed (%s) — continuing without.", exc)
        return None, None, time.monotonic()


def stop_telem(
    thread: Thread | None, event: Event | None, t0: float,
    config: ExpConfig, stage: str,
) -> None:
    """Stop the telemetry sampler and write a cost JSON summary.

    Always logs wall-clock elapsed time, even when the sampler thread
    never started (in which case ``thread``/``event`` are ``None``).
    Never raises.
    """
    elapsed = time.monotonic() - t0
    log.info("Stage '%s' wall-clock: %.1f s (%.2f min)", stage, elapsed, elapsed / 60)
    if thread is None or event is None:
        return
    # Under torchrun-launched DDP every rank runs this clean-up.  Gate the
    # cost JSON write on rank 0 so we don't get N copies of cost_<stage>.json
    # racing the same path.
    from core.dist_util import is_rank_zero
    if not is_rank_zero():
        return
    try:
        out_path = stop_sampler(thread, event)
        rate = float(os.environ.get("VASTAI_RATE_USD_HR", _VASTAI_RATE_USD_HR))
        cost = cost_summarise(out_path, rate)
        cost_path = os.path.join(config.output_dir, f"cost_{stage}.json")
        with open(cost_path, "w") as fh:
            json.dump(cost, fh, indent=2)
        log.info("Cost summary → %s", cost_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("Telemetry stop/cost failed (%s) — continuing.", exc)
