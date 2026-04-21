"""Train DONUT on SROIE with all 13 silent F1-destroying bugs prevented.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: fine-tunes the VisionEncoderDecoder DONUT (~200M params) with Bug 1
    (lm_head dedup), Bug 2 (decoder_start_token_id), Bug 7 (kwargs leak),
    and Bug 9 (stale generation_config) guardrails.  Uses differential LR.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from core.types import DataSplit, ExpConfig
from models.donut_dataset import _DonutCollator, _seed_worker, _SROIEDataset
from models.donut_optim import _make_compute_metrics, _split_param_groups

try:
    import torch
    from transformers import (
        DonutProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
    )
except ImportError:  # lightweight CI — torch/transformers not installed
    pass

__all__ = [
    "_DonutCollator",
    "_SROIEDataset",
    "_make_compute_metrics",
    "_split_param_groups",
    "train_donut",
]


def _prepare_model(config: ExpConfig) -> tuple[Any, Any]:
    """Load DONUT, add SROIE tokens, apply Bug 1/2/7/9 fixes."""
    proc: DonutProcessor = DonutProcessor.from_pretrained(config.base_model)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        config.base_model,
    )
    proc.tokenizer.add_special_tokens({"additional_special_tokens": config.new_tokens})
    # Bug 1: untie lm_head BEFORE resize so that resize_token_embeddings does
    # not create a shared-storage alias between embed_tokens and lm_head.
    # Setting the flag on BOTH top-level and decoder sub-config is required
    # because VisionEncoderDecoderModel reads the sub-config when deciding
    # whether to re-tie weights on resize.
    model.config.tie_word_embeddings = False
    model.config.decoder.tie_word_embeddings = False
    model.decoder.resize_token_embeddings(len(proc.tokenizer), mean_resizing=False)
    # Clone unconditionally so even if the model tied weights internally during
    # resize, lm_head.weight is now a distinct tensor with a unique data_ptr();
    # safetensors identifies duplicates by data_ptr() and would otherwise drop
    # lm_head, producing "missing keys: ['decoder.lm_head.weight']" on reload.
    # The clone MUST happen before the optimizer is built so the optimizer
    # tracks this exact Parameter for the whole of training.
    model.decoder.lm_head.weight = torch.nn.Parameter(
        model.decoder.lm_head.weight.data.clone(),
    )
    model.config.decoder_start_token_id = proc.tokenizer.convert_tokens_to_ids(
        ["<s_sroie>"],
    )[0]  # Bug 2
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.eos_token_id = proc.tokenizer.convert_tokens_to_ids(
        ["</s_sroie>"],
    )[0]
    model.config.vocab_size = model.config.decoder.vocab_size
    # Bug 9: Seq2SeqTrainer's predict_with_generate reads generation_config
    # (snapshotted at from_pretrained time), NOT live model.config. Without
    # mirroring our overrides, eval-time generation starts from the stale
    # donut-base mBART ``<s>`` and token2json returns ``{}`` for every sample
    # → eval_f1 ≡ 0.0 while eval_loss drops normally.
    gc = model.generation_config
    gc.decoder_start_token_id = model.config.decoder_start_token_id
    gc.eos_token_id = model.config.eos_token_id
    gc.pad_token_id = model.config.pad_token_id
    gc.bos_token_id = model.config.decoder_start_token_id
    gc.forced_bos_token_id = None
    gc.forced_eos_token_id = None
    w, h = config.image_size
    model.config.encoder.image_size = [h, w]
    proc.image_processor.size = {"height": h, "width": w}
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    # Bug 7: transformers ≥4.48 adds num_items_in_batch to model inputs when
    # forward has **kwargs (VisionEncoderDecoderModel does). That leaks into
    # kwargs_encoder, forwarded to SwinModel.forward() which has no **kwargs
    # → TypeError on the first training batch. ``accepts_loss_kwargs=False``
    # tells the Trainer to skip this path.
    model.accepts_loss_kwargs = False
    return proc, model


def _build_args(config: ExpConfig, out_dir: str) -> Seq2SeqTrainingArguments:
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and config.precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and config.precision == "fp16"  # Bug 4: only when explicit
    return Seq2SeqTrainingArguments(
        output_dir=out_dir, num_train_epochs=config.epochs_donut,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr, lr_scheduler_type=config.lr_scheduler_type,
        # Issue 3: HF Trainer uses warmup_steps when it is > 0, overriding
        # warmup_ratio. Pass 0 when warmup_ratio is configured so the
        # ratio-based schedule takes effect.
        warmup_ratio=config.warmup_ratio,
        warmup_steps=0 if config.warmup_ratio > 0 else config.warmup_steps,
        weight_decay=config.weight_decay, bf16=use_bf16, fp16=use_fp16,
        max_grad_norm=config.max_grad_norm,
        label_smoothing_factor=config.label_smoothing,
        gradient_checkpointing=config.gradient_checkpointing,
        save_strategy="epoch", eval_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1", greater_is_better=True,
        predict_with_generate=True, generation_num_beams=config.num_beams,
        generation_max_length=config.max_length,
        dataloader_num_workers=2, seed=config.seed, data_seed=config.seed,
    )


def train_donut(config: ExpConfig, data: DataSplit) -> str:
    """Train DONUT with differential LR; return saved model directory."""
    proc, model = _prepare_model(config)
    out_dir = os.path.join(config.output_dir, "donut")
    args = _build_args(config, out_dir)
    # Differential LR: encoder=lr, decoder (incl. resized embeddings + lm_head)
    # =lr_decoder. Pass the optimizer pre-built; scheduler stays None so the
    # Trainer still applies warmup_ratio / cosine per param group.
    optimizer = torch.optim.AdamW(
        _split_param_groups(model, lr_encoder=config.lr, lr_decoder=config.lr_decoder),
        weight_decay=config.weight_decay,
    )
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=_SROIEDataset(data.train, proc, config),
        eval_dataset=_SROIEDataset(data.val, proc, config),
        data_collator=_DonutCollator(model),
        compute_metrics=_make_compute_metrics(proc, config.fields),
        optimizers=(optimizer, None),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
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
