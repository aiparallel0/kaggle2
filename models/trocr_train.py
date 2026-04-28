"""Train TrOCR-small-printed on SROIE crops for text transcription.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: fine-tunes TrOCR with Bug 1 (lm_head dedup), Bug 6 (epochs < 5 guard),
    Bug 7 (kwargs leak), and Bug 9 (stale generation_config) guardrails.
    Token-F1 eval drives load_best_model_at_end.
"""
from __future__ import annotations

import os
from typing import Any

from core.errors import TrainError
from core.metrics import token_f1
from core.types import Crop, ExpConfig
from models.gen_config import _persist_generation_config

_import_error: ImportError | None = None
try:
    import numpy as np
    import torch
    from transformers import (
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrOCRProcessor,
        VisionEncoderDecoderModel,
    )

    _DATASET_BASE: type = torch.utils.data.Dataset
except ImportError as _exc:  # lightweight CI — torch/transformers not installed
    _import_error = _exc
    _DATASET_BASE = object


class _CropDataset(_DATASET_BASE):  # type: ignore[misc]
    def __init__(
        self,
        crops: list[Crop],
        processor: TrOCRProcessor,
        config: ExpConfig,
    ) -> None:
        self._crops = crops
        self._processor = processor
        self._config = config

    def __len__(self) -> int:
        return len(self._crops)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        crop = self._crops[idx]
        from PIL import Image
        img = Image.open(crop.image_path).convert("RGB")
        # Crop to bbox (normalised x1, y1, x2, y2)
        w, h = img.size
        x1, y1, x2, y2 = crop.bbox
        region = img.crop((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
        if region.width < 1 or region.height < 1:
            region = img
        pv = self._processor(images=region, return_tensors="pt").pixel_values.squeeze(0)
        tok = self._processor.tokenizer(
            crop.text,
            max_length=self._config.trocr_max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tok.input_ids.squeeze(0)
        labels[labels == self._processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels}


def train_trocr(config: ExpConfig, crops: list[Crop]) -> str:
    """Fine-tune TrOCR on SROIE crops; return saved model directory."""
    # Bug 6 (gate): TrOCR needs ≥ 5 epochs; guard off disables the floor.
    if config.bug_flags.get("bug_6", True) and config.epochs_trocr < 5:
        raise TrainError(
            f"epochs_trocr={config.epochs_trocr} < 5 — "
            "TrOCR will produce empty outputs (Bug 6)."
        )
    if not crops:
        raise TrainError("No crops provided to train_trocr — check YOLO output.")
    if _import_error is not None:
        raise ImportError(
            "torch and transformers are required for TrOCR training. "
            "Run: pip install -r requirements.txt"
        ) from _import_error

    processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(config.trocr_model)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        config.trocr_model
    )
    # Bug 1: untie output_projection so safetensors doesn't dedup/drop it
    model.config.tie_word_embeddings = False
    model.config.decoder.tie_word_embeddings = False
    if hasattr(model.decoder, "output_projection"):
        model.decoder.output_projection.weight = torch.nn.Parameter(
            model.decoder.output_projection.weight.data.clone()
        )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # Bug 9: mirror ids into generation_config so saved config doesn't overwrite
    gc = model.generation_config
    gc.decoder_start_token_id = model.config.decoder_start_token_id
    gc.eos_token_id = model.config.eos_token_id
    gc.pad_token_id = model.config.pad_token_id
    gc.bos_token_id = model.config.decoder_start_token_id
    # Bug 7: transformers ≥4.48 leaks num_items_in_batch to encoder → TypeError
    model.accepts_loss_kwargs = False

    out_dir = os.path.join(config.output_dir, "trocr")
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and config.precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and config.precision == "fp16"  # Bug 4: only when explicit
    # Shuffle deterministically so val split is uncorrelated with filename order
    import random as _random
    shuffled = list(crops)
    _random.Random(config.seed).shuffle(shuffled)
    split = int(len(shuffled) * 0.9)
    train_crops, val_crops = shuffled[:split], shuffled[split:]
    if not val_crops:
        val_crops = shuffled[:1]

    pad = processor.tokenizer.pad_token_id

    def _compute_metrics(pred: Any) -> dict[str, float]:
        """Per-crop token-F1 (OCR-level, **not** KIE-level).

        Returned under the key ``crop_cer_f1`` — HuggingFace Trainer
        prepends ``eval_`` when logging, yielding ``eval_crop_cer_f1``
        in ``training_log.json`` / trainer state.  The previous
        ``f1`` key mislead every downstream reader (including the
        paper's Table I writer) into treating the 0.96 per-crop OCR
        F1 as a field-level KIE F1; the downstream YOLO-detect /
        Assigner-assign stages stack three further failure modes on
        top of this number, so the KIE F1 is always far lower.
        """
        preds, labels = pred.predictions, pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels == -100, pad, labels)
        preds = np.where(preds == -100, pad, preds)
        p_txt = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        g_txt = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = [token_f1(g, p) for g, p in zip(g_txt, p_txt, strict=True)]
        return {"crop_cer_f1": float(sum(scores) / len(scores)) if scores else 0.0}

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config.epochs_trocr,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        bf16=use_bf16,
        fp16=use_fp16,
        max_grad_norm=config.max_grad_norm,  # Bug 4
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_crop_cer_f1",
        greater_is_better=True,
        predict_with_generate=True,
        seed=config.seed,
    )
    train_ds = _CropDataset(train_crops, processor, config)
    val_ds = _CropDataset(val_crops, processor, config)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=_compute_metrics,
    )
    trainer.train()
    # Bug 9: _load_best_model restores the best checkpoint in-place, losing the
    # patched token ids.  Re-pin BEFORE save_model so the first on-disk write is
    # already correct.  _persist_generation_config also asserts the round-trip.
    _persist_generation_config(
        model, out_dir,
        processor.tokenizer.cls_token_id,
        processor.tokenizer.sep_token_id,
        processor.tokenizer.pad_token_id,
    )
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)  # belt-and-braces
    return out_dir
