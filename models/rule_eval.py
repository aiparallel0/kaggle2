"""Rule-based eval over SROIE GT-OCR text stream (no HF/GPU dependency).

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: evaluates rule_based_assign on SROIE box/ annotations using the same
    rule_based_assign pathway as the live pipeline, but bypassing YOLO+TrOCR
    by feeding ground-truth (GT) OCR text/boxes directly as the OCR stream.
    This GT-OCR-stream baseline isolates the contribution of YOLO/TrOCR OCR
    quality — any gap vs. the full pipeline is attributable to OCR noise.
"""
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from core.metrics import compute_metrics
from core.types import EvalBundle, ExpConfig, Field, Metrics, Prediction, Receipt
from data.sroie_crops import _parse_box_file
from models.donut_eval import normalize_total
from models.pipeline_normalize import (
    normalize_address,
    normalize_company,
    normalize_date,
)
from models.rule_based import rule_based_assign


def _receipt_to_regions(
    receipt: Receipt, fields: list[str],
) -> tuple[list[str], list[list[float]]]:
    """Parse SROIE box file into (texts, bboxes) sorted top-to-bottom."""
    crops = _parse_box_file(receipt, fields)
    if not crops:
        return [], []
    crops.sort(key=lambda c: c.bbox[1])
    texts = [c.text for c in crops]
    bboxes = [list(c.bbox) for c in crops]
    return texts, bboxes


def eval_gtocr_rulebased(config: ExpConfig, test: list[Receipt]) -> Metrics:
    """Run rule_based_assign on GT-OCR text stream; write metrics JSON.

    Mirrors the YOLO+TrOCR+Regex pipeline interface but bypasses YOLO and
    TrOCR by feeding SROIE ground-truth box text/bboxes directly as the OCR
    stream.  The bboxes are normalised [x1,y1,x2,y2] — the same coordinate
    convention produced by ``models/pipeline_detect._detect_and_read``.
    This GT-OCR-stream baseline lets us compare the hybrid pipeline output
    against the same rule_based_assign logic running on perfect OCR input.
    """
    predictions: list[Prediction] = []
    for rec in test:
        texts, bboxes = _receipt_to_regions(rec, config.fields)
        assigned = rule_based_assign(texts, bboxes) if texts else {}
        predictions.append(Prediction(
            receipt_id=rec.image_path.stem,
            fields=[Field(name=k, value=v) for k, v in assigned.items()],
        ))
    # Symmetric per-field normalization — matches ``eval_donut`` /
    # ``eval_pipeline`` so gtocr_rulebased_f1 is comparable across systems
    # (same ``RM 43.50`` == ``43.50`` semantics; same ``SDN BHD.`` ==
    # ``SDN BHD`` semantics for company/address).
    _norms: dict[str, Callable[[str], str]] = {
        "total": normalize_total, "date": normalize_date,
        "company": normalize_company, "address": normalize_address,
    }

    def _identity(s: str) -> str:
        return s

    def _nt(fs: list[Field]) -> list[Field]:
        return [Field(name=f.name,
                      value=_norms.get(f.name.lower(), _identity)(f.value))
                for f in fs]
    n_preds = [Prediction(receipt_id=p.receipt_id, fields=_nt(p.fields))
               for p in predictions]
    n_test = [Receipt(image_path=r.image_path, fields=_nt(r.fields)) for r in test]
    metrics = compute_metrics(EvalBundle(
        predictions=n_preds, receipts=n_test, fields=config.fields,
    ))
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "gtocr_rulebased_metrics.json", "w") as f:
        json.dump({
            "global_f1": metrics.global_f1,
            "global_ned": metrics.global_ned,
            "global_em": metrics.global_em,
            "per_field_f1": metrics.per_field_f1,
            "per_field_ned": metrics.per_field_ned,
            "per_field_em": metrics.per_field_em,
            "n_test": len(test),
            "note": (
                "Rule-based assignment over SROIE GT-OCR text stream "
                "(ground-truth box/text fed directly; no YOLO/TrOCR inference). "
                "Serves as a fair baseline: same rule_based_assign logic as the "
                "live pipeline, but with perfect OCR input."
            ),
        }, f, indent=2)
    return metrics


def _empty_field_map(fields: list[str]) -> dict[str, float]:
    """Zero-initialized per-field dict (placeholder generation)."""
    return dict.fromkeys(fields, 0.0)


def combined_from_rulebased(
    config: ExpConfig, gtocr_rulebased: Metrics,
) -> dict[str, object]:
    """Build paper-injection dict when DONUT/pipeline unavailable (CPU-only)."""
    zero_fields = _empty_field_map(config.fields)
    return {
        "donut_f1": 0.0, "donut_ned": 0.0, "donut_em": 0.0,
        "pipeline_f1": 0.0, "pipeline_ned": 0.0, "pipeline_em": 0.0,
        "rulebased_f1": gtocr_rulebased.global_f1,
        "rulebased_ned": gtocr_rulebased.global_ned,
        "gtocr_rulebased_f1": gtocr_rulebased.global_f1,
        "gtocr_rulebased_ned": gtocr_rulebased.global_ned,
        "gtocr_rulebased_em": gtocr_rulebased.global_em,
        "f1_gap": 0.0,
        "assigner_delta": 0.0,
        "donut_f1_company": zero_fields["company"],
        "donut_f1_date": zero_fields["date"],
        "donut_f1_address": zero_fields["address"],
        "donut_f1_total": zero_fields["total"],
        "epochs_donut": config.epochs_donut,
        "epochs_trocr": config.epochs_trocr,
        "epochs_yolo": config.epochs_yolo,
        "epochs_assigner": config.epochs_assigner,
        "batch_size": config.batch_size,
        "lr": config.lr,
        "lr_encoder": config.lr,
        "lr_decoder": config.lr_decoder,
        "precision": config.precision,
        "label_smoothing": config.label_smoothing,
        "warmup_steps": config.warmup_steps,
        "yolo_img_size": config.yolo_img_size,
        "img_w": config.image_size[0],
        "img_h": config.image_size[1],
        "kd_attn_weight": config.kd_attn_weight,
        "kd_logits_weight": config.kd_logits_weight,
        # Pipeline-diagnostic placeholders: real values are merged from
        # pipeline_metrics.json by stage_paper when it exists.  Zeros
        # here mean "no pipeline run executed in this artefact".
        "empty_detection_fraction": 0.0,
        "per_receipt_error_fraction": 0.0,
        "parity_ok": True,
        # Assigner-telemetry placeholders: real values merged by
        # stage_paper from assigner_metrics.json when available.
        "assigner_params_k": 0.0,
        "assigner_best_epoch": 0,
        "assigner_stopped_at": 0,
        "assigner_best_val_loss": 0.0,
        "artifact_mode": "gtocr_rulebased_only",
    }


def _field_key(field: str, metric: str) -> str:
    return f"rulebased_{metric}_{field}"


def per_field_injection(metrics: Metrics) -> dict[str, float]:
    """Expose per-field rule-based F1/NED as individual \\VAR{} keys."""
    out: dict[str, float] = {}
    for f, v in metrics.per_field_f1.items():
        out[_field_key(f, "f1")] = v
    for f, v in metrics.per_field_ned.items():
        out[_field_key(f, "ned")] = v
    return out
