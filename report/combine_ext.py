"""Extended merge helpers — fold new diagnostics sidecars into the paper dict.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: :mod:`report.combine` caps at 161 LOC and already hosts the
    original three merge helpers (``merge_assigner_metrics``,
    ``merge_pipeline_diagnostics``, ``merge_cost_json``).  This sibling
    module adds the six new merge helpers required by the Section-B
    metric expansion so the per-file cap stays honoured.  Every helper
    has the same 2-in/1-out signature — ``(config, metrics_dict)`` →
    None, mutating in place — and never raises on missing files: a
    flaky train run simply leaves some ``\\VAR{}`` placeholders
    resolving to the ``---`` backstop.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from core.types import ExpConfig

log = logging.getLogger("kaggle2")


def _load_json(path: str | Path) -> dict[str, object] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        with p.open() as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("combine_ext: cannot read %s (%s)", p, exc)
        return None
    return obj if isinstance(obj, dict) else None


def _flatten(prefix: str, data: dict[str, object], target: dict[str, object]) -> None:
    """Fold a flat diagnostics dict into ``target`` with ``<prefix>_<key>``."""
    for k, v in data.items():
        if k == "schema_version":
            continue
        if isinstance(v, dict):
            # Two-level dict (e.g. per_class_ap, cer_per_field) expands
            # as ``prefix_key_subkey`` so the LaTeX injector can address
            # e.g. \VAR{yolo_per_class_ap_total}.
            for sk, sv in v.items():
                target[f"{prefix}_{k}_{sk}"] = sv
        else:
            target[f"{prefix}_{k}"] = v


def merge_yolo_diagnostics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/yolo_metrics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "yolo_metrics.json")
    data = _load_json(path) or _load_json(os.path.join(config.output_dir, "yolo_metrics.json"))
    if data is not None:
        _flatten("yolo", data, metrics)


def merge_trocr_diagnostics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/trocr_metrics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "trocr_metrics.json")
    data = _load_json(path) or _load_json(os.path.join(config.output_dir, "trocr_metrics.json"))
    if data is not None:
        _flatten("trocr", data, metrics)


def merge_assigner_diag(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/assigner_diagnostics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "assigner_diagnostics.json")
    data = _load_json(path) or _load_json(
        os.path.join(config.output_dir, "assigner_diagnostics.json"),
    )
    if data is not None:
        _flatten("assigner", data, metrics)


def merge_donut_diag(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/donut_diagnostics.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "metrics", "donut_diagnostics.json")
    data = _load_json(path) or _load_json(
        os.path.join(config.output_dir, "donut_diagnostics.json"),
    )
    if data is not None:
        _flatten("donut", data, metrics)


def merge_latency(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold every ``metrics/latency_<system>.json`` into the paper dict."""
    metrics_dir = Path(config.output_dir) / "metrics"
    roots: list[Path] = []
    if metrics_dir.is_dir():
        roots.extend(metrics_dir.glob("latency_*.json"))
    # Back-compat path (pre-metrics-subdir runs).
    roots.extend(Path(config.output_dir).glob("latency_*.json"))
    for p in roots:
        system = p.stem.removeprefix("latency_")
        data = _load_json(p)
        if data is not None:
            _flatten(f"{system}_latency", data, metrics)


def merge_extended_metrics(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``metrics/extended_metrics.json`` (per-field CIs, P/R) into paper."""
    path = os.path.join(config.output_dir, "metrics", "extended_metrics.json")
    data = _load_json(path) or _load_json(
        os.path.join(config.output_dir, "extended_metrics.json"),
    )
    if not data:
        return
    # ``extended_metrics.json`` is already namespaced by system key
    # (``donut_*``, ``pipeline_*``, ``rulebased_*``); no prefix added.
    for k, v in data.items():
        metrics.setdefault(k, v)


def merge_env(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold ``env/hostinfo.json`` into the paper metrics dict."""
    path = os.path.join(config.output_dir, "env", "hostinfo.json")
    data = _load_json(path)
    if data is None:
        return
    for k, v in data.items():
        if k == "host" and isinstance(v, dict):
            for sk, sv in v.items():
                metrics.setdefault(f"host_{sk}", sv)
        elif k != "schema_version":
            metrics.setdefault(f"env_{k}", v)


def merge_ablations(config: ExpConfig, metrics: dict[str, object]) -> None:
    """Fold optional ``metrics/ablations.json`` into the paper dict.

    Ablation runs write a flat ``{"ablation_<tag>_<key>": value, ...}``
    JSON; the helper just forwards every entry into ``metrics``.
    """
    path = os.path.join(config.output_dir, "metrics", "ablations.json")
    data = _load_json(path)
    if data is None:
        return
    for k, v in data.items():
        metrics.setdefault(k, v)
