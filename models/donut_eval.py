"""Evaluate DONUT on SROIE test split → Metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import ExpConfig, Field, Metrics, Prediction, Receipt

try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
except ImportError:  # lightweight CI — torch/transformers not installed
    pass


def _token2json_safe(processor: DonutProcessor, tokens: str) -> dict[str, str]:
    """Bug 3 fix: token2json may return a list on CORD data; merge into dict.

    When the list contains multiple values for the same key (happens on
    multi-line addresses decoded as separate pages), prefer the *longest*
    non-empty value — short strings are almost always truncations.
    """
    result = processor.token2json(tokens)
    if isinstance(result, list):
        merged: dict[str, str] = {}
        for page in result:
            if not isinstance(page, dict):
                continue
            for k, v in page.items():
                sv = str(v)
                if k not in merged or len(sv) > len(merged[k]):
                    merged[k] = sv
        return merged
    if isinstance(result, dict):
        return {k: str(v) for k, v in result.items()}
    return {}


def eval_donut(
    model_path: str, test: list[Receipt], config: ExpConfig | None = None,
) -> Metrics:
    """Run DONUT inference on test receipts; return Metrics."""
    if not Path(model_path).exists():
        raise EvalError(f"DONUT model not found at {model_path}")
    processor: DonutProcessor = DonutProcessor.from_pretrained(model_path)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        model_path,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
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
    num_beams = config.num_beams if config is not None else 4
    max_len = config.max_length if config is not None else 768
    predictions: list[Prediction] = []
    from PIL import Image
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            pv = processor(images=img, return_tensors="pt", legacy=False).pixel_values.to(device)
            out = model.generate(
                pv,
                decoder_start_token_id=start_id,
                eos_token_id=eos_id,
                max_length=max_len,
                num_beams=num_beams,
                early_stopping=True,
            )
            tokens = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = _token2json_safe(processor, tokens)
            fields = [Field(name=k, value=v) for k, v in parsed.items()]
            rid = rec.image_path.stem
            predictions.append(Prediction(receipt_id=rid, fields=fields))
    fields_list = ["company", "date", "address", "total"]
    metrics = compute_metrics(predictions, test, fields_list)
    out_dir = os.path.dirname(model_path)
    with open(os.path.join(out_dir, "donut_metrics.json"), "w") as f:
        json.dump(
            {
                "global_f1": metrics.global_f1,
                "global_ned": metrics.global_ned,
                "global_em": metrics.global_em,
                "per_field_f1": metrics.per_field_f1,
            },
            f, indent=2,
        )
    return metrics
