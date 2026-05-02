"""Attention-faithfulness measurement for the FOCUS-T learned assigner.

Project: kaggle2 — Document KIE.
Closes ``docs/HONESTY.md §2.7`` — the prior paper claimed the assigner's
attention map is "directly interpretable" without quantifying that
claim against any standard interpretability measure.  This module
adds the two canonical measurements:

* **Deletion AUC** — ablate the top-k attention-ranked lines; measure
  the prediction-correctness drop.  Higher AUC = the attention map
  faithfully identifies the lines the model relies on (Samek et al.,
  Evaluating the Visualization of What a Deep Neural Network Has
  Learned, IEEE TNNLS 2017, https://arxiv.org/abs/1509.06321).

* **Insertion AUC** — start from an empty receipt; insert lines in
  attention-rank order and measure when correctness recovers.
  Higher AUC = high-attention lines genuinely carry the predictive
  signal (Petsiuk et al., *RISE: Randomized Input Sampling for
  Explanation of Black-box Models*, BMVC 2018,
  https://arxiv.org/abs/1806.07421).

Both metrics are field-conditioned: we report deletion-AUC and
insertion-AUC per field so the ``address`` head's faithfulness
(which spreads attention over a multi-line span) is reported
separately from the ``total`` head's (which peaks).  The aggregate
macro-AUC pair populates the paper's interpretability claim.

Usage (called from ``stages.eval`` when
``config.measure_attention_faithfulness=True``):

    from models.attention_faithfulness import measure_faithfulness
    fa = measure_faithfulness(receipts, attention_samples, predict_fn)
    # fa: {field -> {"deletion_auc": float, "insertion_auc": float}}
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger("kaggle2.faithfulness")

__all__ = [
    "FaithfulnessResult",
    "deletion_auc",
    "insertion_auc",
    "measure_faithfulness",
    "save_faithfulness_report",
]


@dataclass
class FaithfulnessResult:
    """Per-field faithfulness numbers + the underlying curves.

    ``curve`` is the per-step prediction-correctness vector — element
    ``[k]`` is the predicted-string-equals-gold flag after the
    top-``k`` lines have been ablated (deletion) or inserted (insertion).
    """

    field: str
    deletion_auc: float
    insertion_auc: float
    deletion_curve: list[float]
    insertion_curve: list[float]
    n_receipts: int

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "deletion_auc": round(self.deletion_auc, 4),
            "insertion_auc": round(self.insertion_auc, 4),
            "deletion_curve": [round(v, 4) for v in self.deletion_curve],
            "insertion_curve": [round(v, 4) for v in self.insertion_curve],
            "n_receipts": self.n_receipts,
        }


def _correct(prediction: str, gold: str) -> int:
    """Per-receipt exact-match correctness used by both AUCs.

    Whitespace-collapsed casefold equality.  Same shape as the
    ``EM`` (exact-match) component of the headline metric, so the
    faithfulness curves are interpretable in the same units as the
    headline F1 in Sec.~5.
    """
    return int((prediction or "").strip().casefold()
               == (gold or "").strip().casefold())


def _trapezoid_auc(values: list[float]) -> float:
    """Normalised AUC of a 0..1 step curve sampled at integer indices.

    Returns 0.0 for an empty / single-point curve.  Uses the
    composite-trapezoid rule on the ``len(values)-1`` panels.
    """
    if len(values) < 2:
        return 0.0
    width = float(len(values) - 1)
    integral = 0.0
    for a, b in zip(values[:-1], values[1:], strict=False):
        integral += 0.5 * (a + b)
    return integral / width


def deletion_auc(
    rank_indices: list[int],
    predict_with_subset: Callable[[set[int]], str],
    gold: str,
    full_set: set[int],
) -> tuple[float, list[float]]:
    """Deletion-AUC: progressively ablate top-attention lines.

    ``rank_indices`` lists line indices in *attention-rank order*
    (highest-attention first).  ``predict_with_subset`` is a closure
    that, given a *retained* line-index set, returns the predicted
    field string under that masked input.  At step k we retain
    ``full_set - rank_indices[:k]`` and record correctness; the
    AUC of the resulting curve is the deletion-faithfulness number.

    Lower AUC = removing top-attention lines collapses correctness
    quickly = the attention map is faithful (the model genuinely
    relied on those lines).
    """
    curve: list[float] = []
    for k in range(len(rank_indices) + 1):
        retained = full_set - set(rank_indices[:k])
        pred = predict_with_subset(retained)
        curve.append(float(_correct(pred, gold)))
    return _trapezoid_auc(curve), curve


def insertion_auc(
    rank_indices: list[int],
    predict_with_subset: Callable[[set[int]], str],
    gold: str,
) -> tuple[float, list[float]]:
    """Insertion-AUC: progressively reveal top-attention lines.

    Symmetric to deletion: at step k we predict on ``rank_indices[:k]``
    only.  Higher AUC = correctness recovers fast as high-attention
    lines are added = the attention map is faithful.
    """
    curve: list[float] = []
    for k in range(len(rank_indices) + 1):
        retained = set(rank_indices[:k])
        pred = predict_with_subset(retained)
        curve.append(float(_correct(pred, gold)))
    return _trapezoid_auc(curve), curve


def measure_faithfulness(
    receipts: list[object],
    attention_per_receipt: dict[str, dict[str, list[float]]],
    predict_fn: Callable[[str, str, set[int]], str],
    fields: tuple[str, ...] = ("company", "date", "address", "total"),
) -> dict[str, FaithfulnessResult]:
    """Compute deletion+insertion AUC per field across the receipt set.

    Inputs:
        receipts: list of objects with ``image_path`` and per-field
            gold strings (the ``Receipt`` shape from ``core.types``).
        attention_per_receipt: nested map
            ``{receipt_id: {field: [attn_per_line ...]}}`` — the same
            attention rows the assigner emitted at inference time
            (loaded from ``runs/<run_id>/attention_samples.npz``).
        predict_fn: ``(receipt_id, field, retained_indices) -> pred_str``
            closure that runs the FOCUS-T head over the masked input
            and returns the predicted field value.  Caller wires this
            through ``models.eval_pipeline`` so the masking is on the
            same line-set the original prediction saw.

    Returns:
        ``{field -> FaithfulnessResult}`` — per-field deletion AUC,
        insertion AUC, and underlying curves.

    Aggregation note.  AUCs are per-receipt then averaged across the
    test set; curves are the receipt-mean at each step (lengths
    capped at the smallest receipt's line count so the array shape
    stays rectangular).
    """
    per_field_dauc: dict[str, list[float]] = {f: [] for f in fields}
    per_field_iauc: dict[str, list[float]] = {f: [] for f in fields}
    per_field_dcurve: dict[str, list[list[float]]] = {f: [] for f in fields}
    per_field_icurve: dict[str, list[list[float]]] = {f: [] for f in fields}
    n_used = 0

    for r in receipts:
        rid = Path(getattr(r, "image_path", "")).stem
        attn = attention_per_receipt.get(rid)
        if not attn:
            continue
        gold_by_field = {
            f.name: f.value for f in getattr(r, "fields", [])
        }
        n_used += 1
        for fld in fields:
            if fld not in attn:
                continue
            row = attn[fld]
            n_lines = len(row)
            if n_lines == 0:
                continue
            full = set(range(n_lines))
            ranks = sorted(range(n_lines), key=lambda i: -row[i])
            gold = gold_by_field.get(fld, "")
            def _predict(s: set[int], _r: str = rid, _f: str = fld) -> str:
                return predict_fn(_r, _f, s)

            d_auc, d_curve = deletion_auc(ranks, _predict, gold, full)
            i_auc, i_curve = insertion_auc(ranks, _predict, gold)
            per_field_dauc[fld].append(d_auc)
            per_field_iauc[fld].append(i_auc)
            per_field_dcurve[fld].append(d_curve)
            per_field_icurve[fld].append(i_curve)

    out: dict[str, FaithfulnessResult] = {}
    for fld in fields:
        if not per_field_dauc[fld]:
            out[fld] = FaithfulnessResult(
                field=fld, deletion_auc=0.0, insertion_auc=0.0,
                deletion_curve=[], insertion_curve=[], n_receipts=0,
            )
            continue
        # Pad curves to max length and average element-wise.
        max_d = max(len(c) for c in per_field_dcurve[fld])
        max_i = max(len(c) for c in per_field_icurve[fld])
        d_pad = [c + [c[-1]] * (max_d - len(c)) for c in per_field_dcurve[fld]]
        i_pad = [c + [c[-1]] * (max_i - len(c)) for c in per_field_icurve[fld]]
        d_mean = np.mean(np.array(d_pad), axis=0).tolist()
        i_mean = np.mean(np.array(i_pad), axis=0).tolist()
        out[fld] = FaithfulnessResult(
            field=fld,
            deletion_auc=float(np.mean(per_field_dauc[fld])),
            insertion_auc=float(np.mean(per_field_iauc[fld])),
            deletion_curve=d_mean,
            insertion_curve=i_mean,
            n_receipts=len(per_field_dauc[fld]),
        )
    log.info(
        "attention_faithfulness: measured on %d receipts × %d fields",
        n_used, len(fields),
    )
    return out


def save_faithfulness_report(
    results: dict[str, FaithfulnessResult], out_path: Path,
) -> None:
    """Persist the per-field AUC + curves to a JSON sidecar.

    The paper renders this into Table~``\\ref{tab:faithfulness}`` and
    Figure~``\\ref{fig:faithfulness_curves}``.  Aggregate macro-AUCs
    appear under the keys ``faithfulness_deletion_auc_macro`` and
    ``faithfulness_insertion_auc_macro`` for the ``\\VAR{}`` resolver.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body: dict[str, object] = {
        f: r.to_dict() for f, r in results.items()
    }
    if results:
        macro_d = float(np.mean([r.deletion_auc for r in results.values()]))
        macro_i = float(np.mean([r.insertion_auc for r in results.values()]))
        body["faithfulness_deletion_auc_macro"] = round(macro_d, 4)
        body["faithfulness_insertion_auc_macro"] = round(macro_i, 4)
    out_path.write_text(json.dumps(body, indent=2))
    log.info("attention_faithfulness: wrote report to %s", out_path)
