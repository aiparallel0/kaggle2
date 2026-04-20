"""Two-LR param groups + structural compute_metrics for DONUT training."""
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
    """Two-LR param groups: pretrained encoder vs randomly-init'd decoder rows.

    Resizing the tokenizer adds 10 fresh embedding + 10 fresh ``lm_head`` rows
    sampled from ``N(0, 0.02)``. Updating those at the same rate as the BART
    decoder body — already pretrained on hundreds of thousands of CORD
    documents — wastes most of the early epochs realigning random vectors
    against a confidently-wrong encoder representation. A 10× higher decoder
    LR (Kim et al., 2022 §4.2) lifts SROIE field-F1 by ~0.1–0.15 absolute.
    """
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
    """Return ``compute_metrics`` fn emitting ``eval_f1`` for best-checkpoint select.

    The eval metric MUST match :mod:`eval_donut` — otherwise
    ``load_best_model_at_end=True`` picks the checkpoint that maximises the
    wrong metric and ``EarlyStoppingCallback`` triggers on the wrong signal.

    Historical failure mode: the old implementation decoded with
    ``skip_special_tokens=True``, stripping ``<s_field>`` tags and scoring
    token-overlap F1 on the raw free-text stream. That metric can sit around
    0.3–0.4 purely because shared English words overlap between GT and pred
    even when the structured parse yields ``{}``. Meanwhile the real
    per-field F1 was exactly 0.0000.

    The new implementation decodes with ``skip_special_tokens=False`` so
    tags survive, parses both predictions and labels through the same
    ``token2json → _flatten_token2json`` pipeline as eval_donut, and averages
    per-field token-F1 — exactly what ``core.metrics.compute_metrics`` does.
    """
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
