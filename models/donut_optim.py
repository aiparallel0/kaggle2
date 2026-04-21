"""Differential-LR param groups and structural compute_metrics for DONUT.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: implements the 10× decoder LR (Kim et al. 2022 §4.2) that lets resized
    special-token embeddings adapt quickly while the pretrained Swin encoder
    stays stable.  The structural compute_metrics parses both preds and GT
    through token2json so eval_f1 matches compute_metrics exactly.
"""
from __future__ import annotations

from typing import Any

from core.metrics import token_f1
from models.donut_parse import _flatten_token2json

try:
    import numpy as np
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
except ImportError:  # lightweight CI — torch/transformers not installed
    pass


def _split_param_groups(
    model: VisionEncoderDecoderModel, lr_encoder: float, lr_decoder: float,
) -> list[dict[str, Any]]:
    """Differential LR: encoder=lr, decoder=10× lr (resized embeddings adapt fast)."""
    enc: list[torch.nn.Parameter] = []
    dec: list[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # ``decoder.*`` covers the BartDecoder body, embed_tokens (incl. resized
        # rows), and lm_head; everything else is the Swin encoder.
        (dec if n.startswith("decoder.") else enc).append(p)
    return [
        {"params": enc, "lr": lr_encoder},
        {"params": dec, "lr": lr_decoder},
    ]


def _make_compute_metrics(processor: DonutProcessor, fields: list[str]) -> Any:
    """Build compute_metrics that parses preds/GT structurally → eval_f1."""
    pad = processor.tokenizer.pad_token_id

    def _decode_structural(batch: np.ndarray) -> list[str]:
        """Truncate each row at first pad, keep structural tags for parsing."""
        out: list[str] = []
        for row in batch:
            mask = row == pad
            row = row[:int(mask.argmax())] if mask.any() else row
            out.append(processor.tokenizer.decode(row, skip_special_tokens=False))
        return out

    def _parse(tokens_str: str) -> dict[str, str]:
        return _flatten_token2json(processor.token2json(tokens_str))

    def _compute(pred: Any) -> dict[str, float]:
        preds, labels = pred.predictions, pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        # predict_with_generate=True always gives 2-D generated token IDs.
        # The 3-D logit path (teacher-forced argmax) is deliberately omitted:
        # argmax over raw logits inflates early-epoch F1 and masks convergence.
        labels = np.where(labels == -100, pad, labels)
        preds = np.where(preds == -100, pad, preds)
        p_txts = _decode_structural(preds)
        g_txts = _decode_structural(labels)
        f1s: list[float] = []
        for p_txt, g_txt in zip(p_txts, g_txts, strict=True):
            p_fields = _parse(p_txt)
            g_fields = _parse(g_txt)
            per_field = [
                token_f1(
                    g_fields.get(f, "").lower(), p_fields.get(f, "").lower(),
                )
                for f in fields
            ]
            f1s.append(sum(per_field) / len(per_field) if per_field else 0.0)
        return {"f1": float(sum(f1s) / len(f1s)) if f1s else 0.0}

    return _compute
