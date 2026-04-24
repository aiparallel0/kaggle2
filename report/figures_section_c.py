"""Section-C figure orchestrator — drives every new emitter in one call.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: split out of :mod:`stages.paper` so the 166-LOC cap is respected.
    Never raises — any failure in a single emitter is logged and the
    loop continues, guaranteeing the paper stage always produces *some*
    PDF even on a partial run.  Each emitter itself early-returns on
    missing data (see ``figures_common.guard_empty``).
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("kaggle2")


from collections.abc import Callable


def render_section_c(run_dir: Path) -> None:
    """Call every new Section-C figure emitter, swallowing each exception.

    The import chain is deliberately lazy inside the try block so that
    environments without matplotlib (CI, lightweight eval-only
    machines) never fail at module-load time.  The failure log at
    WARNING is sufficient for operators to triage post-hoc.
    """
    try:
        from report.figures_assigner import render_assigner
        from report.figures_calibration import render_calibration
        from report.figures_cost import render_cost
        from report.figures_curves import render_all_curves
        from report.figures_errors import render_errors
        from report.figures_f1 import render_f1_grouped
        from report.figures_gpu import render_gpu_series
        from report.figures_latency import render_latency
        from report.figures_samples import render_samples
        from report.figures_trocr import render_trocr
        from report.figures_yolo import render_yolo
    except ImportError as exc:
        log.warning("Section-C figure emitters unavailable (%s) — skipping.", exc)
        return
    emitters: tuple[tuple[str, Callable[[], object]], ...] = (
        ("curves", lambda: render_all_curves(run_dir)),
        ("f1_grouped", lambda: render_f1_grouped(run_dir)),
        ("calibration", lambda: render_calibration(run_dir)),
        ("latency", lambda: render_latency(run_dir)),
        ("cost", lambda: render_cost(run_dir)),
        ("errors", lambda: render_errors(run_dir)),
        ("yolo", lambda: render_yolo(run_dir)),
        ("trocr", lambda: render_trocr(run_dir)),
        ("assigner", lambda: render_assigner(run_dir)),
        ("gpu_series", lambda: render_gpu_series(run_dir)),
        ("samples", lambda: render_samples(run_dir)),
    )
    for name, fn in emitters:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            log.warning("Section-C figure %s failed (%s) — continuing.", name, exc)
