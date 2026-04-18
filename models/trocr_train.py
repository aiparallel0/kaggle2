"""Train TrOCR on SROIE crops for text transcription."""
from __future__ import annotations

import os
from typing import Any

from core.errors import TrainError
from core.metrics import token_f1
from core.types import Crop, ExpConfig

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
except ImportError:  # lightweight CI — torch/transformers not installed
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
    """Fine-tune TrOCR on receipt crops; return path to saved model."""
    # Bug 6: TrOCR needs >= 5 epochs; config.py enforces this at load time
    if config.epochs_trocr < 5:
        raise TrainError(
            f"epochs_trocr={config.epochs_trocr} < 5 — "
            "TrOCR will produce empty outputs (Bug 6)."
        )
    if not crops:
        raise TrainError("No crops provided to train_trocr — check YOLO output.")

    processor: TrOCRProcessor = TrOCRProcessor.from_pretrained(config.trocr_model)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        config.trocr_model
    )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # Bug 7: transformers ≥4.48 forwards num_items_in_batch into model inputs
    # when forward() has **kwargs; VisionEncoderDecoderModel then passes it to
    # the encoder (SwinModel / ViT) which has no **kwargs → TypeError.
    model.accepts_loss_kwargs = False

    out_dir = os.path.join(config.output_dir, "trocr")
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and config.precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and config.precision == "fp16"  # Bug 4 fix: only enable when explicitly configured
    # Bug 9: ``crops[:split], crops[split:]`` carves the validation set out of
    # the *last* 10 % of the input order, which is itself sorted by receipt
    # filename.  That means val is dominated by receipts whose stems begin
    # with high-numbered prefixes — a non-representative slice that biases
    # eval_f1 by ~0.05 and routinely picks a worse "best" checkpoint.
    # Shuffle deterministically (Random(seed)) so the split is uncorrelated
    # with filename ordering but still reproducible.
    import random as _random
    shuffled = list(crops)
    _random.Random(config.seed).shuffle(shuffled)
    split = int(len(shuffled) * 0.9)
    train_crops, val_crops = shuffled[:split], shuffled[split:]
    if not val_crops:
        val_crops = shuffled[:1]

    pad = processor.tokenizer.pad_token_id

    def _compute_metrics(pred: Any) -> dict[str, float]:
        """Token-F1 over decoded crop texts — drives load_best_model_at_end."""
        preds, labels = pred.predictions, pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels == -100, pad, labels)
        preds = np.where(preds == -100, pad, preds)
        p_txt = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        g_txt = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        scores = [token_f1(g, p) for g, p in zip(g_txt, p_txt, strict=True)]
        return {"f1": float(sum(scores) / len(scores)) if scores else 0.0}

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
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",
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
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    return out_dir
