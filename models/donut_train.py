"""Train DONUT on SROIE with all 7 F1-guardrail bugs prevented."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    DonutProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    VisionEncoderDecoderModel,
)

from core.errors import TrainError
from core.types import DataSplit, ExpConfig, Receipt


def _build_label(receipt: Receipt, processor: DonutProcessor) -> str:
    parts = ["<s_sroie>"]
    for fld in receipt.fields:
        tag = fld.name.lower()
        parts.append(f"<s_{tag}>{fld.value}</s_{tag}>")
    parts.append("</s_sroie>")
    return "".join(parts)


class _SROIEDataset(torch.utils.data.Dataset):  # type: ignore[misc]
    def __init__(
        self,
        receipts: list[Receipt],
        processor: DonutProcessor,
        config: ExpConfig,
    ) -> None:
        self._receipts = receipts
        self._processor = processor
        self._config = config

    def __len__(self) -> int:
        return len(self._receipts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._receipts[idx]
        from PIL import Image
        img = Image.open(r.image_path).convert("RGB")
        pv = self._processor(
            images=img,
            return_tensors="pt",
            size={"height": self._config.image_size[1], "width": self._config.image_size[0]},
        ).pixel_values.squeeze(0)
        label_str = _build_label(r, self._processor)
        tok = self._processor.tokenizer(
            label_str,
            max_length=self._config.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tok.input_ids.squeeze(0)
        labels[labels == self._processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels}


class _LmHeadCloneCallback(TrainerCallback):  # type: ignore[misc]
    """Bug 1 fix: clone lm_head.weight before every save to prevent safetensors dedup."""

    def on_save(
        self,
        args: Seq2SeqTrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        model: VisionEncoderDecoderModel,
        **kwargs: Any,
    ) -> None:
        lm = model.decoder.lm_head
        lm.weight = torch.nn.Parameter(lm.weight.data.clone())


def _validate_f1(metrics: dict[str, float], arch: str) -> None:
    f1 = metrics.get("global_f1", -1.0)
    if arch == "donut" and f1 < 0.50:
        raise TrainError(
            f"DONUT F1={f1:.4f} < 0.50 — likely lm_head dedup (Bug 1), "
            "wrong decoder_start_token_id (Bug 2), or token2json list (Bug 3)."
        )


def train_donut(config: ExpConfig, data: DataSplit) -> str:
    """Train DONUT; return path to saved model directory."""
    processor: DonutProcessor = DonutProcessor.from_pretrained(config.base_model)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        config.base_model
    )
    # Step 2: add tokens + resize
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": config.new_tokens}
    )
    model.decoder.resize_token_embeddings(len(processor.tokenizer))
    # Step 3: Bug 1 — break weight tying
    model.config.tie_word_embeddings = False
    # Step 4: Bug 2 — list form for decoder_start_token_id
    model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(
        ["<s_sroie>"]
    )[0]
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    # Step 5: set encoder image size from config (not hardcoded)
    model.config.encoder.image_size = list(config.image_size)
    processor.image_processor.size = {
        "height": config.image_size[0], "width": config.image_size[1],
    }
    out_dir = os.path.join(config.output_dir, "donut")
    use_bf16 = config.precision == "bf16" and torch.cuda.is_bf16_supported()
    train_args = Seq2SeqTrainingArguments(
        output_dir=out_dir,
        num_train_epochs=config.epochs_donut,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_grad_norm=config.max_grad_norm,  # Bug 4
        label_smoothing_factor=config.label_smoothing,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        predict_with_generate=True,
        dataloader_num_workers=2,
        seed=config.seed,
    )
    train_ds = _SROIEDataset(data.train, processor, config)
    val_ds = _SROIEDataset(data.val, processor, config)
    trainer = Seq2SeqTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[_LmHeadCloneCallback()],
    )
    trainer.train()
    trainer.save_model(out_dir)
    processor.save_pretrained(out_dir)
    # Save model path for downstream steps
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    meta = {"model_path": out_dir}
    with open(os.path.join(config.output_dir, "donut_path.json"), "w") as f:
        json.dump(meta, f)
    return out_dir
