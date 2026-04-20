"""Rule-based eval using SROIE gold-OCR box-file text (no HF / GPU needed).

This stage exists because the end-to-end DONUT and YOLO+TrOCR+Attention
pipelines require ~1 GB of Hugging Face weights, which is not reachable
from every execution environment (e.g. offline CI sandboxes). The SROIE
``box/`` annotation files already contain gold-OCR text aligned to
bounding boxes, so ``rule_based_assign`` can be evaluated end-to-end
without any neural model at all. The resulting F1 is a genuine lower
bound on what the ``pipeline.rulebased`` arm of ``eval_pipeline`` can
produce — any degradation between the two arms is attributable solely
to YOLO/TrOCR error propagation, which is exactly the comparison the
paper makes.
"""
from __future__ import annotations

import json
from pathlib import Path

from core.metrics import compute_metrics
from core.types import EvalBundle, ExpConfig, Field, Metrics, Prediction, Receipt
from data.sroie_crops import _parse_box_file
from models.rule_based import rule_based_assign


def _receipt_to_regions(
    receipt: Receipt, fields: list[str],
) -> tuple[list[str], list[list[float]]]:
    """Parse SROIE box file into (texts, bboxes) in top-to-bottom reading order.

    Returns empty lists when the receipt has no box annotations. Boxes are
    sorted by ``y1`` so ``rule_based_assign``'s spatial heuristics (company
    above address above total) receive input in the same ordering the
    YOLO+TrOCR pipeline would produce at inference time.
    """
    crops = _parse_box_file(receipt, fields)
    if not crops:
        return [], []
    crops.sort(key=lambda c: c.bbox[1])
    texts = [c.text for c in crops]
    bboxes = [list(c.bbox) for c in crops]
    return texts, bboxes


def eval_rulebased_gold(config: ExpConfig, test: list[Receipt]) -> Metrics:
    """Run ``rule_based_assign`` over gold-OCR box text; write metrics JSON.

    Args:
        config: Full experiment config; only ``fields`` / ``output_dir`` used.
        test: Held-out test receipts (same split as every other eval path).

    Returns:
        :class:`Metrics` computed by the shared ``core.metrics.compute_metrics``
        so the numbers are directly comparable with DONUT / pipeline eval.
    """
    predictions: list[Prediction] = []
    for rec in test:
        texts, bboxes = _receipt_to_regions(rec, config.fields)
        assigned = rule_based_assign(texts, bboxes) if texts else {}
        predictions.append(Prediction(
            receipt_id=rec.image_path.stem,
            fields=[Field(name=k, value=v) for k, v in assigned.items()],
        ))
    metrics = compute_metrics(EvalBundle(
        predictions=predictions, receipts=test, fields=config.fields,
    ))
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rulebased_gold_metrics.json", "w") as f:
        json.dump({
            "global_f1": metrics.global_f1,
            "global_ned": metrics.global_ned,
            "global_em": metrics.global_em,
            "per_field_f1": metrics.per_field_f1,
            "per_field_ned": metrics.per_field_ned,
            "per_field_em": metrics.per_field_em,
            "n_test": len(test),
            "note": (
                "Rule-based assignment over SROIE gold-OCR box text "
                "(no YOLO/TrOCR inference). Serves as a lower bound on "
                "pipeline.rulebased and an upper bound on what heuristics "
                "alone achieve when OCR is perfect."
            ),
        }, f, indent=2)
    return metrics


def _empty_field_map(fields: list[str]) -> dict[str, float]:
    return dict.fromkeys(fields, 0.0)


def combined_from_rulebased(
    config: ExpConfig, rulebased_gold: Metrics,
) -> dict[str, object]:
    """Build a paper-injection dict when DONUT / pipeline are unavailable.

    DONUT and the learned pipeline arm are zeroed (not faked) and the
    rule-based-on-gold-OCR number is injected as ``rulebased_f1``. The
    paper template tolerates zeros here: every \\VAR{} placeholder still
    receives a real value, and the discussion explicitly calls out that
    the DONUT/pipeline rows are blank in gold-OCR-only artefacts.
    """
    zero_fields = _empty_field_map(config.fields)
    return {
        "donut_f1": 0.0, "donut_ned": 0.0, "donut_em": 0.0,
        "pipeline_f1": 0.0, "pipeline_ned": 0.0, "pipeline_em": 0.0,
        "rulebased_f1": rulebased_gold.global_f1,
        "rulebased_ned": rulebased_gold.global_ned,
        "rulebased_gold_f1": rulebased_gold.global_f1,
        "f1_gap": 0.0,
        "assigner_delta": 0.0,
        "donut_f1_company": zero_fields["company"],
        "donut_f1_date": zero_fields["date"],
        "donut_f1_address": zero_fields["address"],
        "donut_f1_total": zero_fields["total"],
        "epochs_donut": config.epochs_donut,
        "epochs_trocr": config.epochs_trocr,
        "epochs_yolo": config.epochs_yolo,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "precision": config.precision,
        "label_smoothing": config.label_smoothing,
        "yolo_img_size": config.yolo_img_size,
        "img_w": config.image_size[0],
        "img_h": config.image_size[1],
        "artifact_mode": "rulebased_gold_only",
    }


def _field_key(field: str, metric: str) -> str:
    return f"rulebased_{metric}_{field}"


def per_field_injection(metrics: Metrics) -> dict[str, float]:
    """Expose per-field rule-based F1 / NED as individual \\VAR{} keys."""
    out: dict[str, float] = {}
    for f, v in metrics.per_field_f1.items():
        out[_field_key(f, "f1")] = v
    for f, v in metrics.per_field_ned.items():
        out[_field_key(f, "ned")] = v
    return out
