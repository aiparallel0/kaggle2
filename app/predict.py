"""DONUT single-image inference for the demo server.

Loads the fine-tuned checkpoint from ``{config.output_dir}/donut`` when it
exists, otherwise falls back to ``config.base_model`` so the demo is
runnable *before* training finishes. All the Bug-2/3/9 guards from
``models.donut_eval`` are reused verbatim.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import ExpConfig
from models.donut_parse import token2json_safe

log = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """Loaded DONUT + processor + runtime metadata."""

    processor: Any
    model: Any
    device: str
    start_id: int
    eos_id: int | None
    source: str  # "finetuned" | "base"
    model_path: str


def _resolve_path(config: ExpConfig) -> tuple[str, str]:
    """Return (path, source) — prefer fine-tuned checkpoint, fall back to base."""
    finetuned = os.path.join(config.output_dir, "donut")
    if Path(finetuned).exists() and any(Path(finetuned).iterdir()):
        return finetuned, "finetuned"
    log.warning(
        "No fine-tuned DONUT at %s; falling back to base model %r. "
        "Train with `make train` for meaningful predictions.",
        finetuned, config.base_model,
    )
    return config.base_model, "base"


def load_model(config: ExpConfig) -> LoadedModel:
    """Load DONUT, add SROIE tokens if missing, resolve start/eos ids. 2-in/1-out."""
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel

    path, source = _resolve_path(config)
    processor = DonutProcessor.from_pretrained(path)
    model = VisionEncoderDecoderModel.from_pretrained(path)

    if source == "base":
        added = processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": config.new_tokens},
        )
        if added:
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            model.config.tie_word_embeddings = False  # Bug 1

    # Bug 9: avoid inherited mBART forcing.
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    unk = processor.tokenizer.unk_token_id
    start_id = processor.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]  # Bug 2
    if start_id in (None, unk):
        cfg_start = model.config.decoder_start_token_id
        start_id = int(cfg_start) if cfg_start is not None else 0
    eos_id_raw = processor.tokenizer.convert_tokens_to_ids(["</s_sroie>"])[0]
    eos_id = None if eos_id_raw in (None, unk) else int(eos_id_raw)
    if eos_id is None and model.config.eos_token_id is not None:
        eos_id = int(model.config.eos_token_id)

    return LoadedModel(
        processor=processor, model=model, device=device,
        start_id=int(start_id), eos_id=eos_id,
        source=source, model_path=path,
    )


def predict(loaded: LoadedModel, image: Any) -> dict[str, str]:
    """Run DONUT on a PIL image; return flat ``{field: value}`` dict. 2-in/1-out."""
    import torch

    size_kwargs = {"size": {"height": 960, "width": 1280}}
    with torch.no_grad():
        pv = loaded.processor(
            images=image.convert("RGB"), return_tensors="pt",
            legacy=False, **size_kwargs,
        ).pixel_values.to(loaded.device)
        out = loaded.model.generate(
            pv,
            decoder_start_token_id=loaded.start_id,
            eos_token_id=loaded.eos_id,
            max_length=768, num_beams=4, early_stopping=True,
        )
    tokens = loaded.processor.batch_decode(out, skip_special_tokens=False)[0]
    return token2json_safe(loaded.processor, tokens)
