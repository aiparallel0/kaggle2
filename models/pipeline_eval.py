"""Evaluate YOLOv8 + TrOCR + AttentionAssigner pipeline on SROIE test.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: runs the three-stage pipeline (detect → read → assign) and produces
    PipelineResult containing both learned-assigner and rule-based metrics.
    Samples cross-attention tensors for fig_attention_heatmap.
"""
from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import (
    EvalBundle,
    ExpConfig,
    Field,
    PipelinePaths,
    PipelineResult,
    Prediction,
    Receipt,
)
from models.attention_assign import _load_assigner
from models.donut_eval import normalize_total
from models.pipeline_assign import _assign_learned_with_attn
from models.pipeline_attn import DEFAULT_SAMPLE_K, AttentionSampler
from models.pipeline_consensus import refine_assignments
from models.pipeline_detect import _detect_and_read, _fallback_full_image
from models.pipeline_miss_tracker import log_field_breakdown
from models.pipeline_normalize import (
    normalize_address,
    normalize_company,
    normalize_date,
)
from models.rule_based import rule_based_assign

_import_error: ImportError | None = None
try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError as _exc:  # lightweight CI — torch not installed
    _import_error = _exc

log = logging.getLogger("kaggle2")


_FIELD_NORMALISERS: dict[str, Callable[[str], str]] = {
    "total": normalize_total,
    "date": normalize_date,
    "company": normalize_company,
    "address": normalize_address,
}


def _identity(s: str) -> str:
    return s


def _nt(fields: list[Field]) -> list[Field]:
    """Apply symmetric per-field normalisation before metric compute.

    Every field (not just TOTAL) is routed through its paired
    ``normalize_*`` so pred/GT punctuation/spacing mismatches — which
    token-F1 treats as full token losses — cancel symmetrically on both
    sides.  This mirrors what the ANLS-style metric reported by the
    ICDAR SROIE evaluator does and keeps pipeline F1 comparable to
    DONUT F1 (eval_donut passes through the same normalisers).
    """
    return [Field(
        name=f.name,
        value=_FIELD_NORMALISERS.get(f.name.lower(), _identity)(f.value),
    ) for f in fields]


def _predictions_by_field(
    preds: list[Prediction], gts: list[Receipt], fields: tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    """Per-receipt (pred, gt) pairs for selected fields — triage JSON."""
    gt_map = {r.image_path.stem: {f.name.lower(): f.value for f in r.fields}
              for r in gts}
    return {fn: [{"receipt_id": p.receipt_id,
                  "pred": next((f.value for f in p.fields
                                if f.name.lower() == fn), ""),
                  "gt": gt_map.get(p.receipt_id, {}).get(fn, "")}
                 for p in preds] for fn in fields}


def _paths_from_config(config: ExpConfig) -> PipelinePaths:
    """Derive pipeline checkpoint paths from ``config.output_dir``."""
    return PipelinePaths(
        yolo=os.path.join(config.output_dir, "yolo", "run", "weights", "best.pt"),
        trocr=os.path.join(config.output_dir, "trocr"),
        assigner=os.path.join(config.output_dir, "assigner.pt"),
    )


def _load(paths: PipelinePaths, config: ExpConfig) -> tuple[Any, Any, Any, Any, str]:
    for p, name in [
        (paths.yolo, "YOLO"), (paths.trocr, "TrOCR"), (paths.assigner, "Assigner"),
    ]:
        if not Path(p).exists():
            raise EvalError(f"{name} checkpoint not found at {p}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise EvalError("ultralytics not installed") from exc
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    yolo = YOLO(paths.yolo)
    trocr_proc = TrOCRProcessor.from_pretrained(paths.trocr)
    trocr_model = VisionEncoderDecoderModel.from_pretrained(paths.trocr)
    assigner = _load_assigner(paths.assigner, n_fields=len(config.fields))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trocr_model = trocr_model.to(device)
    assigner = assigner.to(device)
    trocr_model.eval()
    assigner.eval()
    return yolo, trocr_proc, trocr_model, assigner, device


def _resolve_yolo_img(paths: PipelinePaths, config: ExpConfig) -> int:
    meta = Path(paths.assigner).parent / "pipeline_meta.json"
    if meta.exists():
        with open(meta) as f:
            return int(json.load(f).get("yolo_img_size", config.yolo_img_size))
    return config.yolo_img_size


def eval_pipeline(config: ExpConfig, test: list[Receipt]) -> PipelineResult:
    """Run the three-stage pipeline; return assigner + rule-based Metrics."""
    if _import_error is not None:
        raise ImportError(
            "torch is required for pipeline evaluation. "
            "Run: pip install -r requirements.txt"
        ) from _import_error
    paths = _paths_from_config(config)
    yolo, trocr_proc, trocr_model, assigner, device = _load(paths, config)
    yolo_img = _resolve_yolo_img(paths, config)
    preds_l: list[Prediction] = []
    preds_r: list[Prediction] = []
    n_empty_detect = 0  # receipts where YOLO found zero boxes → full-image fallback
    n_receipt_err = 0  # receipts where per-receipt try/except caught a failure
    receipt_error_samples: list[str] = []
    receipt_error_type_set: set[str] = set()
    attn_sampler = AttentionSampler(k=DEFAULT_SAMPLE_K)
    with torch.no_grad():
        if test:  # canary: raises real traceback on systematic shape/device mismatch
            img0 = Image.open(test[0].image_path).convert("RGB")
            t0, f0, b0 = _detect_and_read(yolo, trocr_proc, trocr_model, img0,
                                           str(test[0].image_path), config, yolo_img, device)
            if not t0:
                t0, f0, b0 = _fallback_full_image(trocr_proc, trocr_model, img0, config, device)
            _assign_learned_with_attn(
                assigner, t0, f0, b0, config.fields, device,
                address_accept_fraction=config.address_accept_fraction,
                regex_router=config.regex_router,
                total_confidence_threshold=config.total_confidence_threshold,
            )
        for rec in test:
            rid = rec.image_path.stem
            # Per-receipt isolation: one corrupt scan (OSError), CUDA hiccup
            # (RuntimeError), or bad value (ValueError) must not abort the
            # entire eval run; the receipt scores F1=0 for every field.
            try:
                img = Image.open(rec.image_path).convert("RGB")
                texts, feats, bboxes = _detect_and_read(
                    yolo, trocr_proc, trocr_model, img, str(rec.image_path),
                    config, yolo_img, device,
                )
                # Empty-detect fallback: rare on SROIE but preserves contribution.
                if not texts:
                    n_empty_detect += 1
                    texts, feats, bboxes = _fallback_full_image(
                        trocr_proc, trocr_model, img, config, device,
                    )
                learned, attn = _assign_learned_with_attn(
                    assigner, texts, feats, bboxes, config.fields, device,
                    address_accept_fraction=config.address_accept_fraction,
                    regex_router=config.regex_router,
                    total_confidence_threshold=config.total_confidence_threshold,
                )
                # Analytical per-field refinement: compensates for TrOCR/YOLO
                # mistakes the learned attention alone cannot fix (SUBTOTAL-vs-
                # TOTAL confusion, postcode digit repair, company O↔0 / B↔8,
                # date separator + 8-digit compact reconstruction).
                attn_rows = (
                    [attn[i].tolist() for i in range(attn.shape[0])]
                    if attn is not None else None
                )
                learned = refine_assignments(
                    learned, texts, bboxes, attn_rows, config.fields,
                )
                if attn is not None and not attn_sampler.full:
                    attn_sampler.capture(str(rec.image_path), bboxes, attn)
                rule = rule_based_assign(texts, bboxes) if texts else {}
            except (OSError, RuntimeError, ValueError) as exc:
                log.exception("pipeline_eval: receipt %s failed", rid)
                receipt_error_type_set.add(type(exc).__name__)
                if len(receipt_error_samples) < 3:
                    receipt_error_samples.append(repr(exc))
                n_receipt_err += 1
                learned, rule = {}, {}
            preds_l.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in learned.items()],
            ))
            preds_r.append(Prediction(
                receipt_id=rid,
                fields=[Field(name=k, value=v) for k, v in rule.items()],
            ))
    # Symmetric TOTAL normalisation keeps pipeline F1 comparable to DONUT F1.
    n_preds_l = [Prediction(receipt_id=p.receipt_id, fields=_nt(p.fields))
                 for p in preds_l]
    n_preds_r = [Prediction(receipt_id=p.receipt_id, fields=_nt(p.fields))
                 for p in preds_r]
    n_test = [Receipt(image_path=r.image_path, fields=_nt(r.fields)) for r in test]
    m_l = compute_metrics(EvalBundle(n_preds_l, n_test, config.fields))
    m_r = compute_metrics(EvalBundle(n_preds_r, n_test, config.fields))
    field_diag = log_field_breakdown(m_l, n_preds_l, n_test, config.fields)
    # Flat top-level pipeline_metrics.json so stages/_common,
    # report/combine, and core/validate all read the same file; the attn
    # sampler still writes into results/yolo/ for fig_attn_heatmap.
    out_dir = Path(config.output_dir)
    n_total = max(len(test), 1)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Write attention_samples into ``results/`` (not ``results/yolo/run/``)
    # so :func:`report.figures_attn.render_attention_heatmap` — which
    # reads from ``config.output_dir`` — can find the artefact and
    # render Fig.~\ref{fig:attn_heatmap}.
    attn_sampler.write(out_dir)
    with open(out_dir / "pipeline_metrics.json", "w") as f:
        json.dump({
            "assigner_f1": m_l.global_f1, "rulebased_f1": m_r.global_f1,
            "assigner_ned": m_l.global_ned, "assigner_em": m_l.global_em,
            "per_field_f1": m_l.per_field_f1,
            "rulebased_per_field_f1": m_r.per_field_f1,
            "empty_detection_fraction": n_empty_detect / n_total,
            "per_receipt_error_fraction": n_receipt_err / n_total,
            "n_test_receipts": len(test),
            "receipt_error_samples": receipt_error_samples,
            "receipt_error_types": sorted(receipt_error_type_set),
            "per_field_diagnostics": field_diag,
            "predictions_by_field": _predictions_by_field(
                n_preds_l, n_test, tuple(config.fields),
            ),
        }, f, indent=2)
    return PipelineResult(
        assigner=m_l, rulebased=m_r,
        assigner_preds=n_preds_l, rulebased_preds=n_preds_r,
    )
