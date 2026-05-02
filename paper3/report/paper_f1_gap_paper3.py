"""F1-gap reporter for the Paper 2 / Paper 3 bifurcation.

Project: kaggle2 — FOCUS-$\\Sigma$ verification layer for document KIE.
Role: given two run directories — one produced under
    \\texttt{configs/paper2.json} (rules + zone-prior pipeline) and
    one produced under the Paper-3 preset (learned cross-attention
    assigner + FOCUS-$\\Sigma$ verifier + ensemble heads) — compute
    the headline pipeline-arm F1 gap and persist it to a sidecar.
    Both papers' results sections cite the gap from this sidecar so
    the comparison stays consistent regardless of which paper the
    reader picks up first.

Contract: ``compute_paper_f1_gap(paper2_dir, paper3_dir) -> dict``
    returns a flat scalar dict suitable for forwarding to
    ``\\VAR{}`` keys.  Writes
    ``runs/<paper3_dir>/metrics/paper_f1_gap.json`` so the gap shows
    up in the Paper 3 paper-build under
    ``\\VAR{paper2_pipeline_f1}``,
    ``\\VAR{paper3_pipeline_f1}``,
    ``\\VAR{paper2_paper3_f1_gap}``.

Honest defaults: when either run is missing or its
    ``combined_metrics.json`` lacks the headline pipeline F1, the
    function returns an empty dict and writes nothing — consumers
    fall back to the unresolved-vars audit gate.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("kaggle2")

__all__ = ["compute_paper_f1_gap", "write_paper_f1_gap"]


def _load_pipeline_f1(run_dir: Path) -> float | None:
    """Read ``pipeline_f1`` from a run's ``combined_metrics.json``."""
    candidates = [
        run_dir / "metrics" / "combined_metrics.json",
        run_dir / "combined_metrics.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("paper_f1_gap: cannot read %s (%s).", path, exc)
            continue
        v = data.get("pipeline_f1")
        if isinstance(v, int | float):
            return float(v)
    return None


def compute_paper_f1_gap(
    paper2_dir: Path | str, paper3_dir: Path | str,
) -> dict[str, float]:
    """Return ``{paper2_pipeline_f1, paper3_pipeline_f1, paper2_paper3_f1_gap}``.

    Empty dict when either run's headline pipeline F1 cannot be
    located.  The gap is computed as ``paper3 - paper2`` so a positive
    number means Paper 3 beats Paper 2 — which is the direction the
    bifurcation contract expects.
    """
    p2 = _load_pipeline_f1(Path(paper2_dir))
    p3 = _load_pipeline_f1(Path(paper3_dir))
    if p2 is None or p3 is None:
        return {}
    return {
        "paper2_pipeline_f1": round(p2, 4),
        "paper3_pipeline_f1": round(p3, 4),
        "paper2_paper3_f1_gap": round(p3 - p2, 4),
    }


def write_paper_f1_gap(
    paper2_dir: Path | str, paper3_dir: Path | str,
) -> Path | None:
    """Persist the gap dict to ``<paper3_dir>/metrics/paper_f1_gap.json``.

    Returns the path written, or ``None`` if either run was unreadable.
    Idempotent: re-running overwrites with the freshest numbers.
    """
    out_dir = Path(paper3_dir) / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = compute_paper_f1_gap(paper2_dir, paper3_dir)
    if not payload:
        return None
    out_path = out_dir / "paper_f1_gap.json"
    out_path.write_text(json.dumps(payload, indent=2))
    log.info(
        "paper_f1_gap: paper3 - paper2 = %+.4f (paper2=%.4f, paper3=%.4f)",
        payload["paper2_paper3_f1_gap"],
        payload["paper2_pipeline_f1"],
        payload["paper3_pipeline_f1"],
    )
    return out_path
