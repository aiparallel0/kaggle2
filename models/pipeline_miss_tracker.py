"""Per-field F1 loss tracker — prints where pipeline F1 is being lost.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: produces the per-field F1 breakdown emitted on the console after
    every pipeline eval and the serialised ``per_field_diagnostics``
    block written alongside ``predictions_by_field`` in
    ``pipeline_metrics.json``.  Makes "where is F1 lost?" a one-minute
    question instead of a PR round-trip.
"""
from __future__ import annotations

import logging
from typing import TypedDict

from core.metrics import token_f1
from core.types import Metrics, Prediction, Receipt

log = logging.getLogger("kaggle2")

# Receipts with per-receipt F1 below this threshold are listed as
# "worst offenders" in the console output and in the JSON diagnostics.
_MISS_THRESHOLD = 0.5
# How many worst-offender receipt IDs to list on each console line.
_TOP_MISS_K = 5


class _FieldDiag(TypedDict):
    f1: float
    n_missed: int
    n_total: int
    top_miss_ids: list[str]


def _per_receipt_field_f1(
    preds: list[Prediction], gts: list[Receipt], field: str,
) -> list[tuple[str, float]]:
    """Compute token-F1 per receipt for one field, lower-casing both sides."""
    gt_map: dict[str, str] = {}
    for r in gts:
        rid = r.image_path.stem
        for f in r.fields:
            if f.name.lower() == field:
                gt_map[rid] = f.value.lower()
                break
    out: list[tuple[str, float]] = []
    for p in preds:
        pred_value = ""
        for f in p.fields:
            if f.name.lower() == field:
                pred_value = f.value.lower()
                break
        out.append((p.receipt_id, token_f1(gt_map.get(p.receipt_id, ""), pred_value)))
    return out


def _field_diagnostics(
    preds: list[Prediction], gts: list[Receipt], field: str, field_f1: float,
) -> _FieldDiag:
    """Per-field miss table keyed by receipt_id; used by both log + JSON."""
    per_r = _per_receipt_field_f1(preds, gts, field)
    missed = [(rid, f1) for rid, f1 in per_r if f1 < _MISS_THRESHOLD]
    missed.sort(key=lambda x: x[1])  # worst-first
    return _FieldDiag(
        f1=field_f1,
        n_missed=len(missed),
        n_total=len(per_r),
        top_miss_ids=[rid for rid, _ in missed[:_TOP_MISS_K]],
    )


def log_field_breakdown(
    metrics: Metrics, preds: list[Prediction], gts: list[Receipt],
    fields: list[str],
) -> dict[str, _FieldDiag]:
    """Emit a per-field F1 breakdown to the console and return it as a dict.

    Console format (one line per field, mypy-strict safe):

        Per-field F1 breakdown (pipeline.assigner):
          total    F1=0.5079   misses(F1<0.5)=32/63   top: X51005433549,...
          ...
        Tip: jq -r '.predictions_by_field.total[] | select(.pred!=.gt) |
             [.receipt_id,.pred,.gt] | @tsv' results/pipeline_metrics.json

    Fields are reported in ascending F1 order so the biggest loss source
    is at the top — mirrors the per-field delta table in
    ``assert_pipeline_beats_rulebased_gold``.
    """
    diags: dict[str, _FieldDiag] = {
        f: _field_diagnostics(preds, gts, f, metrics.per_field_f1.get(f, 0.0))
        for f in fields
    }
    order = sorted(fields, key=lambda f: diags[f]["f1"])
    log.info("Per-field F1 breakdown (pipeline.assigner):")
    for f in order:
        d = diags[f]
        top = ",".join(d["top_miss_ids"]) or "-"
        log.info(
            "  %-8s F1=%.4f   misses(F1<%.1f)=%d/%d   top: %s",
            f, d["f1"], _MISS_THRESHOLD, d["n_missed"], d["n_total"], top,
        )
    log.info(
        "Tip: jq -r '.predictions_by_field.%s[] | select(.pred!=.gt) | "
        "[.receipt_id,.pred,.gt] | @tsv' "
        "./results/pipeline_metrics.json | head",
        order[0] if order else "total",
    )
    return diags
