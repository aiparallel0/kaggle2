"""Eval-time producer — writes per-sample preds, error jsonl, extended metrics.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: honest wiring between :mod:`stages.eval` and Section-B sidecars.
    Every helper writes REAL values only — no synthetic fill-ins.  If
    the caller has no ground truth the writer skips the record.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path

from core.metrics_errors import classify_miss
from core.metrics_extended import summarise_extended
from core.types import EvalBundle, Metrics, Prediction, Receipt

log = logging.getLogger("kaggle2")


def _sub(output_dir: str, subdir: str, name: str) -> Path:
    """Resolve ``<output_dir>/<subdir>/<name>`` (auto-create parent)."""
    root = Path(output_dir) / subdir
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def write_preds_jsonl(
    path: Path, predictions: Iterable[Prediction], receipts: Iterable[Receipt],
    fields: tuple[str, ...], model: str,
) -> int:
    """Write one JSON line per (prediction, receipt); return record count."""
    n = 0
    with path.open("w") as fh:
        for pred, rec in zip(predictions, receipts, strict=True):
            gt = {fld.name.lower(): fld.value for fld in rec.fields}
            pr = {fld.name.lower(): fld.value for fld in pred.fields}
            row = {
                "image_id": rec.image_path.stem, "model": model,
                "gt_fields": {f: gt.get(f, "") for f in fields},
                "pred_fields": {f: pr.get(f, "") for f in fields},
                "per_field_exact": {
                    f: gt.get(f, "").strip().lower() == pr.get(f, "").strip().lower()
                    for f in fields
                },
            }
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def write_errors_jsonl(
    path: Path, predictions: Iterable[Prediction], receipts: Iterable[Receipt],
    fields: tuple[str, ...], model: str,
) -> int:
    """Write one row per (image, field) with the 8-category miss label."""
    n = 0
    with path.open("w") as fh:
        for pred, rec in zip(predictions, receipts, strict=True):
            gt = {fld.name.lower(): fld.value for fld in rec.fields}
            pr = {fld.name.lower(): fld.value for fld in pred.fields}
            for f in fields:
                g, p = gt.get(f, ""), pr.get(f, "")
                fh.write(json.dumps({
                    "image_id": rec.image_path.stem, "model": model, "field": f,
                    "gold": g, "pred": p, "category": classify_miss(f, g, p),
                }) + "\n")
                n += 1
    return n


def write_extended_metrics(
    path: Path, bundles: dict[str, tuple[Metrics, EvalBundle]],
    n_iter: int, level: float,
) -> int:
    """Write ``extended_metrics.json`` keyed by ``<system>_<metric>``."""
    out: dict[str, object] = {}
    for system, (metrics, bundle) in bundles.items():
        for k, v in summarise_extended(metrics, bundle, n_iter=n_iter, level=level).items():
            out[f"{system}_{k}"] = v
    with path.open("w") as fh:
        json.dump(out, fh, indent=2)
    return len(out)


def _merge_error_jsonls(output_dir: str, sources: tuple[str, ...]) -> None:
    """Concat ``predictions/<src>.jsonl`` into ``per_field_errors.jsonl``."""
    merged = _sub(output_dir, "predictions", "per_field_errors.jsonl")
    with merged.open("w") as out_fh:
        for name in sources:
            part = Path(output_dir) / "predictions" / name
            if part.is_file():
                out_fh.write(part.read_text())


def _build_bundles(
    fields: tuple[str, ...],
    donut: tuple[list[Prediction] | None, Metrics | None],
    pipe: tuple[list[Prediction] | None, Metrics | None],
    receipts: list[Receipt],
) -> dict[str, tuple[Metrics, EvalBundle]]:
    """Assemble the ``{system: (metrics, bundle)}`` dict for extended metrics."""
    out: dict[str, tuple[Metrics, EvalBundle]] = {}
    for key, (preds, metrics) in {"donut": donut, "pipeline": pipe}.items():
        if preds is not None and metrics is not None and receipts:
            out[key] = (metrics, EvalBundle(
                predictions=preds, receipts=receipts, fields=list(fields),
            ))
    return out


def emit_all(
    output_dir: str, fields: tuple[str, ...], *,
    donut_preds: list[Prediction] | None, pipeline_preds: list[Prediction] | None,
    receipts: list[Receipt],
    donut_metrics: Metrics | None, pipeline_metrics: Metrics | None,
    n_iter: int = 1000, level: float = 0.95,
) -> dict[str, int]:
    """One-call producer.  Writes everything that can be written from real data."""
    counts: dict[str, int] = {}
    for system, preds in (("donut", donut_preds), ("pipeline", pipeline_preds)):
        if preds is None or not receipts:
            continue
        counts[f"{system}_preds"] = write_preds_jsonl(
            _sub(output_dir, "predictions", f"{system}_preds.jsonl"),
            preds, receipts, fields, system,
        )
        counts[f"{system}_errors"] = write_errors_jsonl(
            _sub(output_dir, "predictions", f"{system}_errors.jsonl"),
            preds, receipts, fields, system,
        )
    _merge_error_jsonls(output_dir, ("donut_errors.jsonl", "pipeline_errors.jsonl"))
    bundles = _build_bundles(
        fields, (donut_preds, donut_metrics), (pipeline_preds, pipeline_metrics),
        receipts,
    )
    if bundles:
        counts["extended_keys"] = write_extended_metrics(
            _sub(output_dir, "metrics", "extended_metrics.json"),
            bundles, n_iter=n_iter, level=level,
        )
    log.info("eval_producers: wrote %s", counts)
    return counts
