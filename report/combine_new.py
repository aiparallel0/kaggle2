"""Merge helpers for the P1 / P2 / P4 proposal side-cars.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: :mod:`report.combine_ext` holds the older merge helpers; this
    sibling file hosts only the new-feature producers so each file
    stays under the per-file LOC cap.  Every helper shares the
    ``(config, metrics_dict) -> None`` contract, never raises on a
    missing side-car, and is idempotent when re-invoked.

Side-cars consumed:

  * ``metrics/ablation_report.json`` — P1 13-bug atlas (delta + CI).
  * ``metrics/foundation_metrics.json`` — P4 zero-shot ceiling.
  * ``metrics/rag_ablation.json``       — P2 RAG-on vs RAG-off paired run.
"""
from __future__ import annotations

import os

from core.types import ExpConfig
from report.combine_ext import _load_json


def merge_ablation_report(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold the P1 13-bug ablation report into the paper dict.

    Produces ``\\VAR{bug_<N>_delta}``, ``\\VAR{bug_<N>_ci_low}``,
    ``\\VAR{bug_<N>_ci_high}`` for N in 1..13 plus ``\\VAR{all_off_delta}``
    / ``\\VAR{ablation_baseline_f1}`` / ``\\VAR{ablation_n_seeds}``.
    Missing report → no keys inserted (unresolved \\VAR fallback to ---).
    """
    path = os.path.join(config.output_dir, "metrics", "ablation_report.json")
    data = _load_json(path)
    if data is None:
        return
    baseline = data.get("baseline_f1")
    if isinstance(baseline, int | float):
        metrics.setdefault("ablation_baseline_f1", float(baseline))
    n_seeds = data.get("n_seeds")
    if isinstance(n_seeds, int):
        metrics.setdefault("ablation_n_seeds", n_seeds)
    deltas = data.get("per_bug_delta") or {}
    cis_lo = data.get("per_bug_ci_low") or {}
    cis_hi = data.get("per_bug_ci_high") or {}
    if isinstance(deltas, dict):
        for k, v in deltas.items():
            metrics.setdefault(f"{k}_delta", v)
    if isinstance(cis_lo, dict):
        for k, v in cis_lo.items():
            metrics.setdefault(f"{k}_ci_low", v)
    if isinstance(cis_hi, dict):
        for k, v in cis_hi.items():
            metrics.setdefault(f"{k}_ci_high", v)


def merge_foundation_metrics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold the P4 foundation-oracle side-car into the paper dict.

    Produces ``\\VAR{foundation_f1}``, ``\\VAR{foundation_ned}``,
    ``\\VAR{foundation_em}`` plus per-field variants when present.
    Silently skipped when the foundation arm did not run.
    """
    path = os.path.join(config.output_dir, "metrics", "foundation_metrics.json")
    data = _load_json(path) or _load_json(
        os.path.join(config.output_dir, "foundation_metrics.json"),
    )
    if data is None:
        return
    # Paper uses the ``foundation_`` prefix so figures and tables can
    # overlay the zero-shot ceiling on top of trained-system bars.
    for src, dst in (
        ("global_f1", "foundation_f1"),
        ("global_ned", "foundation_ned"),
        ("global_em", "foundation_em"),
    ):
        if src in data and isinstance(data[src], int | float):
            metrics.setdefault(dst, float(data[src]))
    per_field = data.get("per_field_f1") or {}
    if isinstance(per_field, dict):
        for field_name, value in per_field.items():
            if isinstance(value, int | float):
                metrics.setdefault(f"foundation_f1_{field_name}", float(value))


def merge_rag_metrics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold the P2 RAG-ablation side-car into the paper dict.

    A paired RAG-on / RAG-off run writes
    ``{"rag_on_f1": .., "rag_off_f1": .., "rag_on_ned": .., ...}`` to
    ``metrics/rag_ablation.json``; the helper forwards the keys
    verbatim so the ``experiments.tex`` ablation row resolves directly.
    """
    path = os.path.join(config.output_dir, "metrics", "rag_ablation.json")
    data = _load_json(path)
    if data is None:
        return
    for k, v in data.items():
        if k == "schema_version":
            continue
        metrics.setdefault(k, v)
