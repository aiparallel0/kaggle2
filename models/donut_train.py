"""Train DONUT on SROIE with all 7 F1-guardrail bugs prevented."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from core.metrics import token_f1
from core.types import DataSplit, ExpConfig, Receipt

try:
    import numpy as np
    import torch
    from transformers import (
        DonutProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        TrainerCallback,
        VisionEncoderDecoderModel,
    )

    _DATASET_BASE: type = torch.utils.data.Dataset
    _CALLBACK_BASE: type = TrainerCallback
except ImportError:  # lightweight CI — torch/transformers not installed
    _DATASET_BASE = object
    _CALLBACK_BASE = object


def _build_label(receipt: Receipt) -> str:
    parts = ["<s_sroie>"]
    for fld in receipt.fields:
        t = fld.name.lower()
        parts.append(f"<s_{t}>{fld.value}</s_{t}>")
    parts.append("</s_sroie>")
    return "".join(parts)


class _SROIEDataset(_DATASET_BASE):  # type: ignore[misc]
    def __init__(
        self, receipts: list[Receipt], processor: DonutProcessor, config: ExpConfig,
    ) -> None:
        self._r, self._p, self._c = receipts, processor, config

    def __len__(self) -> int:
        return len(self._r)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        r = self._r[idx]
        from PIL import Image
        img = Image.open(r.image_path).convert("RGB")
        pv = self._p(
            images=img, return_tensors="pt",
            size={"height": self._c.image_size[1], "width": self._c.image_size[0]},
        ).pixel_values.squeeze(0)
        tok = self._p.tokenizer(
            _build_label(r), max_length=self._c.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        input_ids = tok.input_ids.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self._p.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels, "decoder_input_ids": input_ids}


class _LmHeadCloneCallback(_CALLBACK_BASE):  # type: ignore[misc]
    """Bug 1: clone lm_head.weight before every save to defeat safetensors dedup."""

    def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        m = kwargs["model"]
        m.decoder.lm_head.weight = torch.nn.Parameter(
            m.decoder.lm_head.weight.data.clone()
        )


def _seed_worker(_worker_id: int) -> None:
    """Deterministic DataLoader workers — prevents silent nondeterminism."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _make_compute_metrics(processor: DonutProcessor) -> Any:
    """Return compute_metrics fn emitting eval_f1 for best-checkpoint selection."""
    pad = processor.tokenizer.pad_token_id

    def _compute(pred: Any) -> dict[str, float]:
        preds, labels = pred.predictions, pred.label_ids
        if isinstance(preds, tuple):
            preds = preds[0]
        labels = np.where(labels == -100, pad, labels)
        p_txt = processor.tokenizer.batch_decode(preds, skip_special_tokens=True)
        g_txt = processor.tokenizer.batch_decode(labels, skip_special_tokens=True)
        s = [token_f1(g, p) for g, p in zip(g_txt, p_txt, strict=True)]
        return {"f1": float(sum(s) / len(s)) if s else 0.0}

    return _compute


def train_donut(config: ExpConfig, data: DataSplit) -> str:
    """Train DONUT; return path to saved model directory."""
    proc: DonutProcessor = DonutProcessor.from_pretrained(config.base_model)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        config.base_model,
    )
    proc.tokenizer.add_special_tokens({"additional_special_tokens": config.new_tokens})
    model.decoder.resize_token_embeddings(len(proc.tokenizer))
    model.config.tie_word_embeddings = False  # Bug 1
    model.config.decoder_start_token_id = proc.tokenizer.convert_tokens_to_ids(
        ["<s_sroie>"],
    )[0]  # Bug 2
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    w, h = config.image_size
    model.config.encoder.image_size = [h, w]
    proc.image_processor.size = {"height": h, "width": w}
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    out_dir = os.path.join(config.output_dir, "donut")
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and config.precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and not use_bf16  # Bug 4
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, num_train_epochs=config.epochs_donut,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio, warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay, bf16=use_bf16, fp16=use_fp16,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing,
        gradient_checkpointing=config.gradient_checkpointing,
        save_strategy="epoch", eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", greater_is_better=True,
        predict_with_generate=True, generation_num_beams=config.num_beams,
        generation_max_length=config.max_length,
        dataloader_num_workers=2, seed=config.seed, data_seed=config.seed,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=_SROIEDataset(data.train, proc, config),
        eval_dataset=_SROIEDataset(data.val, proc, config),
        compute_metrics=_make_compute_metrics(proc),
        callbacks=[
            _LmHeadCloneCallback(),
            EarlyStoppingCallback(early_stopping_patience=config.patience),
        ],
    )
    _orig = trainer.get_train_dataloader
    gen = torch.Generator().manual_seed(config.seed)

    def _det() -> Any:
        dl = _orig()
        dl.worker_init_fn = _seed_worker
        dl.generator = gen
        return dl

    trainer.get_train_dataloader = _det
    trainer.train()
    trainer.save_model(out_dir)
    proc.save_pretrained(out_dir)
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "donut_path.json"), "w") as f:
        json.dump({"model_path": out_dir}, f)
    return out_dir
