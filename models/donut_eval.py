"""Evaluate DONUT on SROIE test split with all Bug 2/3/8/9 guards.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: runs DONUT inference with decoder_start_token_id=<s_sroie> (Bug 2),
    flattens token2json output (Bug 3/8), disables forced EOS (Bug 9),
    and normalizes TOTAL values for symmetric metric comparison.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import EvalBundle, ExpConfig, Field, Metrics, Prediction, Receipt
from models.donut_parse import token2json_safe as _token2json_safe

try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
except ImportError:  # lightweight CI — torch/transformers not installed
    pass

__all__ = ["eval_donut", "_token2json_safe", "normalize_total"]

_CURRENCY_RE = re.compile(r"(?:\brm|\busd|\bmyr)\.?\s*", re.IGNORECASE)
_CURRENCY_SYMS = ("$", "₹", "€", "£")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(,\d{3})+(\.\d{1,2})?$")
_NUMERIC_TOKEN_RE = re.compile(r"^\d+(\.\d{1,2})?$")


def normalize_total(s: str) -> str:
    """Normalize TOTAL string for symmetric DONUT/GT comparison."""
    if not s:
        return s
    t = _CURRENCY_RE.sub("", s)
    for sym in _CURRENCY_SYMS:
        t = t.replace(sym, "")
    t = t.strip()
    if _THOUSANDS_RE.match(t):
        t = t.replace(",", "")
    tokens = [tok for tok in t.split() if _NUMERIC_TOKEN_RE.match(tok)]
    if tokens:
        t = tokens[-1]
    try:
        return f"{float(t):.2f}"
    except ValueError:
        return t


def _apply_total_normalizer(preds: list[Prediction], gts: list[Receipt]) -> tuple[
    list[Prediction], list[Receipt],
]:
    """Return normalized copies with ``total`` fields passed through
    :func:`normalize_total` — preds and GT both, so the metric is symmetric."""
    def _norm_fields(fs: list[Field]) -> list[Field]:
        out: list[Field] = []
        for fld in fs:
            if fld.name.lower() == "total":
                out.append(Field(name=fld.name, value=normalize_total(fld.value)))
            else:
                out.append(fld)
        return out

    n_preds = [Prediction(receipt_id=p.receipt_id, fields=_norm_fields(p.fields))
               for p in preds]
    n_gts = [Receipt(image_path=r.image_path, fields=_norm_fields(r.fields))
             for r in gts]
    return n_preds, n_gts


def _load(model_path: str) -> tuple[Any, Any, str]:
    if not Path(model_path).exists():
        raise EvalError(f"DONUT model not found at {model_path}")
    processor: DonutProcessor = DonutProcessor.from_pretrained(model_path)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # Bug 9 (eval side): donut-base ships a generation_config with
    # ``forced_eos_token_id`` pointing at mBART's ``</s>`` (id 2). HF applies
    # forcing after picking the next token, so generation emits token 2 at
    # the second-to-last position and our ``</s_sroie>`` is never produced.
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None
    return processor, model, device


def _resolve_start_eos(processor: Any, model: Any) -> tuple[int, int | None]:
    start_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]  # Bug 2
    unk_id = processor.tokenizer.unk_token_id
    if start_id is None or start_id == unk_id:
        cfg_start = model.config.decoder_start_token_id
        if cfg_start is None:
            raise EvalError(
                "decoder_start_token_id unresolved: <s_sroie> is not in the "
                "tokenizer and model.config.decoder_start_token_id is unset.",
            )
        start_id = int(cfg_start)
    eos_id = processor.tokenizer.convert_tokens_to_ids(["</s_sroie>"])[0]
    if eos_id is None or eos_id == unk_id:
        cfg_eos = model.config.eos_token_id
        eos_id = int(cfg_eos) if cfg_eos is not None else None
    return start_id, eos_id


def eval_donut(config: ExpConfig, test: list[Receipt]) -> Metrics:
    """Run DONUT inference on test receipts; return Metrics (2-in/1-out)."""
    model_path = os.path.join(config.output_dir, "donut")
    processor, model, device = _load(model_path)
    start_id, eos_id = _resolve_start_eos(processor, model)
    num_beams = config.num_beams
    max_len = config.max_length
    # Pin inference image size to the training size (stale checkpoints can
    # silently fall back to the model-card default and lose ~0.05 F1).
    size_kwargs: dict[str, Any] = {"size": {
        "height": config.image_size[1], "width": config.image_size[0],
    }}
    predictions: list[Prediction] = []
    from PIL import Image
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            pv = processor(
                images=img, return_tensors="pt", legacy=False, **size_kwargs,
            ).pixel_values.to(device)
            out = model.generate(
                pv, decoder_start_token_id=start_id, eos_token_id=eos_id,
                max_length=max_len, num_beams=num_beams, early_stopping=True,
            )
            tokens = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = _token2json_safe(processor, tokens)
            fields = [Field(name=k, value=v) for k, v in parsed.items()]
            predictions.append(Prediction(receipt_id=rec.image_path.stem, fields=fields))
    # Symmetric numeric normalization for the TOTAL field (DONUT only):
    # matches the paper's "normalized numeric comparison for the Total field"
    # so F1/EM/NED measure semantic match rather than incidental formatting.
    norm_preds, norm_test = _apply_total_normalizer(predictions, test)
    metrics = compute_metrics(EvalBundle(
        predictions=norm_preds, receipts=norm_test, fields=config.fields,
    ))
    out_dir = os.path.dirname(model_path)
    with open(os.path.join(out_dir, "donut_metrics.json"), "w") as f:
        json.dump(
            {
                "global_f1": metrics.global_f1, "global_ned": metrics.global_ned,
                "global_em": metrics.global_em, "per_field_f1": metrics.per_field_f1,
            }, f, indent=2,
        )
    return metrics
