"""Evaluate DONUT on SROIE test split → Metrics."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import ExpConfig, Field, Metrics, Prediction, Receipt

try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
except ImportError:  # lightweight CI — torch/transformers not installed
    pass


def _token2json_safe(processor: DonutProcessor, tokens: str) -> dict[str, str]:
    """Normalise ``processor.token2json`` output to a flat ``{field: value}`` dict.

    Two shapes must be flattened:

    * **Bug 3 — list return (CORD multi-page).** ``token2json`` returns
      ``[{...}, {...}]`` when it sees ``<sep/>`` tokens in the output stream.
      Each page may contain the same key; the longest non-empty string wins
      because short strings are almost always truncations.

    * **Bug 8 — outer ``<s_sroie>`` wrapper.** Our training labels wrap every
      receipt in ``<s_sroie>…</s_sroie>`` (this tag is also the
      ``decoder_start_token_id`` / ``eos_token_id``). HuggingFace's
      ``token2json`` parses the wrapper as a root key, returning
      ``{"sroie": {"company": "X", "date": "Y", …}}``. Downstream
      ``compute_metrics`` looks up ``company`` / ``date`` / ``address`` /
      ``total`` directly on that dict and sees missing keys, driving global
      F1 to exactly ``0.0000`` even when the model decoded the fields
      correctly. Flattening nested dicts collects the real field-level
      entries regardless of how many wrapper layers ``token2json``
      introduces.
    """
    return _flatten_token2json(processor.token2json(tokens))


def _flatten_token2json(obj: Any) -> dict[str, str]:
    """Collect all string-valued leaf entries from a nested dict/list tree.

    Recurses into dict values and list elements so wrapper keys (e.g.
    ``"sroie"``) and CORD-style page lists collapse to a single flat
    ``{field: value}`` mapping. On duplicate keys, the longest value wins,
    matching the Bug-3 fix rationale (address lines are usually truncated
    on the first occurrence).
    """
    merged: dict[str, str] = {}

    def _merge(key: str, value: str) -> None:
        if key not in merged or len(value) > len(merged[key]):
            merged[key] = value

    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, dict | list):
                for sub_k, sub_v in _flatten_token2json(v).items():
                    _merge(sub_k, sub_v)
            else:
                _merge(k, str(v))
    elif isinstance(obj, list):
        for entry in obj:
            if isinstance(entry, dict | list):
                for sub_k, sub_v in _flatten_token2json(entry).items():
                    _merge(sub_k, sub_v)
    return merged


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
    # Pin inference image size to the training size.  DonutProcessor stores
    # ``size`` inside its image_processor and persists it via save_pretrained,
    # but a stale checkpoint or a transformers version that drops the field
    # silently falls back to the model-card default (1280x960 → 2560x1920),
    # which interpolates positional embeddings and degrades F1 by ~0.05.
    # Passing size= per call is the source-of-truth fix.
    if config is not None:
        size_kwargs: dict[str, Any] = {"size": {
            "height": config.image_size[1], "width": config.image_size[0],
        }}
    else:
        size_kwargs = {}
    predictions: list[Prediction] = []
    from PIL import Image
    with torch.no_grad():
        for rec in test:
            img = Image.open(rec.image_path).convert("RGB")
            pv = processor(
                images=img, return_tensors="pt", legacy=False, **size_kwargs,
            ).pixel_values.to(device)
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
    # Use config.fields when available; fall back to defaults only for callers
    # that predate the config parameter (e.g. standalone scripts without a config).
    fields_list = config.fields if config is not None else ["company", "date", "address", "total"]
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
