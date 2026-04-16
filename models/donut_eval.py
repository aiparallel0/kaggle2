"""Evaluate DONUT on SROIE test split → Metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from transformers import DonutProcessor, VisionEncoderDecoderModel

from core.errors import EvalError
from core.types import Field, Metrics, Prediction, Receipt


def _token2json_safe(processor: DonutProcessor, tokens: str) -> dict[str, str]:
    """Bug 3 fix: token2json may return a list on CORD data; merge into dict."""
    result = processor.token2json(tokens)
    if isinstance(result, list):
        merged: dict[str, str] = {}
        for page in result:
            if isinstance(page, dict):
                for k, v in page.items():
                    if k not in merged:
                        merged[k] = str(v)
        return merged
    if isinstance(result, dict):
        return {k: str(v) for k, v in result.items()}
    return {}


def _compute_metrics(
    predictions: list[Prediction],
    receipts: list[Receipt],
    fields: list[str],
) -> Metrics:
    pf1: dict[str, list[float]] = {f: [] for f in fields}
    pned: dict[str, list[float]] = {f: [] for f in fields}
    pem: dict[str, list[float]] = {f: [] for f in fields}
    for pred, rec in zip(predictions, receipts, strict=True):
        gt = {fld.name.lower(): fld.value.lower() for fld in rec.fields}
        pr = {fld.name.lower(): fld.value.lower() for fld in pred.fields}
        for f in fields:
            g = gt.get(f, "")
            p = pr.get(f, "")
            em = 1.0 if g == p else 0.0
            ned = _ned(g, p)
            f1 = _token_f1(g, p)
            pem[f].append(em)
            pned[f].append(ned)
            pf1[f].append(f1)
    per_f1 = {f: sum(v) / len(v) for f, v in pf1.items() if v}
    per_ned = {f: sum(v) / len(v) for f, v in pned.items() if v}
    per_em = {f: sum(v) / len(v) for f, v in pem.items() if v}
    g_f1 = sum(per_f1.values()) / len(per_f1) if per_f1 else 0.0
    g_ned = sum(per_ned.values()) / len(per_ned) if per_ned else 0.0
    g_em = sum(per_em.values()) / len(per_em) if per_em else 0.0
    return Metrics(
        global_f1=g_f1, global_ned=g_ned, global_em=g_em,
        per_field_f1=per_f1, per_field_ned=per_ned, per_field_em=per_em,
    )


def _ned(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    dist = _edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            dp[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _token_f1(a: str, b: str) -> float:
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = ta & tb
    p = len(common) / len(tb)
    r = len(common) / len(ta)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def eval_donut(model_path: str, test: list[Receipt]) -> Metrics:
    """Run DONUT inference on test receipts; return Metrics."""
    if not Path(model_path).exists():
        raise EvalError(f"DONUT model not found at {model_path}")
    processor: DonutProcessor = DonutProcessor.from_pretrained(model_path)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        model_path
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # Bug 2: list-form decoder_start_token_id. Fall back to model.config if
    # <s_sroie> is absent from the reloaded tokenizer (e.g. non-SROIE checkpoint).
    start_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    unk_id = processor.tokenizer.unk_token_id
    if start_id is None or start_id == unk_id:
        cfg_start = model.config.decoder_start_token_id
        if cfg_start is None:
            raise EvalError(
                "decoder_start_token_id unresolved: <s_sroie> is not in the "
                "tokenizer and model.config.decoder_start_token_id is unset."
            )
        start_id = int(cfg_start)
    predictions: list[Prediction] = []
    from PIL import Image
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            pv = processor(images=img, return_tensors="pt").pixel_values.to(device)
            out = model.generate(
                pv,
                decoder_start_token_id=start_id,
                max_length=768,
                early_stopping=True,
            )
            tokens = processor.batch_decode(out, skip_special_tokens=False)[0]
            parsed = _token2json_safe(processor, tokens)
            fields = [Field(name=k, value=v) for k, v in parsed.items()]
            rid = rec.image_path.stem
            predictions.append(Prediction(receipt_id=rid, fields=fields))
    fields_list = ["company", "date", "address", "total"]
    metrics = _compute_metrics(predictions, test, fields_list)
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
