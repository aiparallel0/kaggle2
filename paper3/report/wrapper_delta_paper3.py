"""Wrapper-$\\Delta$ producer for the NeurIPS-variant paper build.

Project: kaggle2 — FOCUS-$\\Sigma$ verification layer for document KIE.
Role: aggregates per-architecture, per-dataset, pre/post FOCUS-$\\Sigma$
    metrics into the flat ``\\VAR{}`` dict consumed by
    ``report/sections/results_neurips.tex`` and
    ``report/sections/intro_neurips.tex``.

Contract: ``merge_wrapper_delta(config, metrics) -> None`` — same
    ``(ExpConfig, dict) -> None`` shape used by every other merge
    helper (see :mod:`report.combine`, :mod:`report.combine_new`).
    Reads sidecars under ``runs/<id>/metrics/`` and forwards their
    keys verbatim; never raises on a missing file; always idempotent.

Honest defaults: when a sidecar is absent the producer leaves the
    corresponding ``\\VAR{}`` keys unresolved so they surface in
    ``metrics/unresolved_vars.json``.  No synthetic numbers are
    fabricated.  The only exception is the small set of provenance /
    config keys (``cord_test_n``, ``layoutlmv3_epochs``,
    ``pipeline_f1_total_baseline``) which are read directly from the
    live ``ExpConfig`` or the already-resolved ``combined_metrics.json``
    headline (no inference / extrapolation).
"""
from __future__ import annotations

import os
from typing import Any

from core.types import ExpConfig
from report.combine_ext import _load_json

__all__ = ["merge_wrapper_delta"]


_WRAPPER_DELTA_SIDECAR = "wrapper_delta_metrics.json"
_ABLATION_SIDECAR = "ablation_focus_sigma.json"
_ERROR_DECOMP_SIDECAR = "error_decomposition.json"
_FAITHFULNESS_SIDECAR = "faithfulness_metrics.json"
_CALIBRATION_SIDECAR = "calibration_metrics.json"
_LATENCY_SIDECAR = "latency_metrics.json"


def merge_wrapper_delta(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold every NeurIPS-variant sidecar into the paper metrics dict.

    Idempotent: every write goes through ``setdefault`` so an explicit
    producer-written value wins.  No keys are synthesised when the
    sidecar is absent — unresolved keys are left for
    ``metrics/unresolved_vars.json`` to enumerate.
    """
    out_metrics_dir = os.path.join(config.output_dir, "metrics")
    _merge_flat_sidecar(out_metrics_dir, _WRAPPER_DELTA_SIDECAR, metrics)
    _merge_flat_sidecar(out_metrics_dir, _ABLATION_SIDECAR, metrics)
    _merge_flat_sidecar(out_metrics_dir, _ERROR_DECOMP_SIDECAR, metrics)
    _merge_flat_sidecar(out_metrics_dir, _FAITHFULNESS_SIDECAR, metrics)
    _merge_flat_sidecar(out_metrics_dir, _CALIBRATION_SIDECAR, metrics)
    _merge_flat_sidecar(out_metrics_dir, _LATENCY_SIDECAR, metrics)
    _merge_config_provenance(config, metrics)
    _heal_baseline_from_headline(metrics)


def _merge_flat_sidecar(
    metrics_dir: str, name: str, metrics: dict[str, object],
) -> None:
    """Forward every ``(key, scalar)`` pair from a sidecar JSON.

    ``schema_version`` and any nested-dict / list values are skipped:
    the LaTeX layer expects flat scalars.  Lists of length 2 are
    forwarded as a paired-bootstrap CI string ``[lo, hi]`` so the
    ``..._ci`` keys read naturally in tables.
    """
    data = _load_json(os.path.join(metrics_dir, name))
    if not isinstance(data, dict):
        return
    for key, value in data.items():
        if key == "schema_version":
            continue
        if isinstance(value, (str, int, float, bool)):
            metrics.setdefault(key, value)
        elif isinstance(value, list) and len(value) == 2 and all(
            isinstance(v, (int, float)) for v in value
        ):
            metrics.setdefault(
                key, f"[{float(value[0]):.4f}, {float(value[1]):.4f}]",
            )


def _merge_config_provenance(
    config: ExpConfig, metrics: dict[str, object],
) -> None:
    """Surface config-level constants the NeurIPS sections cite directly.

    These are not measurements — they are knobs of the run — so they
    can be safely populated from the live ``ExpConfig`` without any
    risk of inventing a number.  Each ``setdefault`` lets a
    measurement-time write (if present in a sidecar) take precedence.
    """
    cord_n = _safe_int(getattr(config, "cord_test_n", None))
    if cord_n is not None:
        metrics.setdefault("cord_test_n", cord_n)
    l3_epochs = _safe_int(getattr(config, "layoutlmv3_epochs", None))
    if l3_epochs is not None:
        metrics.setdefault("layoutlmv3_epochs", l3_epochs)


def _heal_baseline_from_headline(metrics: dict[str, object]) -> None:
    """Carry the bare FOCUS-T total-F1 from the headline run, if present.

    The NeurIPS framing splits ``pipeline_f1_total`` (bare candidate,
    pre-$\\Sigma$) from ``pipeline_f1_total_focus_sigma`` (post-wrap).
    When no wrapper-$\\Delta$ sidecar exists, the headline
    ``combined_metrics.json::pipeline_f1_total`` is by construction the
    *bare* number (FOCUS-$\\Sigma$ Identity-3 is opt-in via
    ``focus_sigma_enabled`` — see ``configs/default.json``).  We
    forward it as ``pipeline_f1_total_baseline`` only — never as the
    post-wrap key — so the two columns of the wrapper-$\\Delta$ table
    stay distinguishable.
    """
    bare = metrics.get("pipeline_f1_total")
    if isinstance(bare, (int, float)):
        metrics.setdefault("pipeline_f1_total_baseline", float(bare))


def _safe_int(value: Any) -> int | None:
    """Coerce optional config attributes to ``int`` without raising."""
    if isinstance(value, bool):
        return None  # bool is-a int in Python but not what we want
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None
