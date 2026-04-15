"""Train TrOCR on SROIE crops for text transcription."""
from __future__ import annotations

import os
from typing import Any

import torch
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from core.errors import TrainError
from core.types import Crop, ExpConfig


class _CropDataset(torch.utils.data.Dataset):  # type: ignore[misc]
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
        pv = self._processor(images=img, return_tensors="pt").pixel_values.squeeze(0)
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

    out_dir = os.path.join(config.output_dir, "trocr")
    use_bf16 = config.precision == "bf16" and torch.cuda.is_bf16_supported()
    split = int(len(crops) * 0.9)
    train_crops, val_crops = crops[:split], crops[split:]
    if not val_crops:
        val_crops = crops[:1]

    args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config.epochs_trocr,
        per_device_train_batch_size=config.batch_size,
        learning_rate=config.lr,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_grad_norm=config.max_grad_norm,  # Bug 4
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        predict_with_generate=True,
        seed=config.seed,
    )
    train_ds = _CropDataset(train_crops, processor, config)
    val_ds = _CropDataset(val_crops, processor, config)
    trainer = Seq2SeqTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    )
    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    return out_dir
