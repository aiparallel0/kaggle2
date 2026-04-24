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
from models._gen_config import _persist_generation_config
from models.donut_dataset import _DonutCollator, _seed_worker, _SROIEDataset
from models.donut_optim import _make_compute_metrics, _split_param_groups

_import_error: ImportError | None = None
try:
    import torch
    from transformers import (
        DonutProcessor,
        EarlyStoppingCallback,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        VisionEncoderDecoderModel,
    )
except ImportError as _exc:  # lightweight CI — torch/transformers not installed
    _import_error = _exc

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
    extra_tokens = list(config.new_tokens)
    # P2 (RAG): add <retrieved>/</retrieved> sentinels alongside SROIE tags
    # so the decoder can attend to the serialised neighbour span prefix.
    if config.rag_enabled:
        for tok in ("<retrieved>", "</retrieved>"):
            if tok not in extra_tokens:
                extra_tokens.append(tok)
    proc.tokenizer.add_special_tokens({"additional_special_tokens": extra_tokens})
    # Bug 1 (gate): untie lm_head BEFORE resize so safetensors cannot
    # dedup the shared-storage alias.  Flag off = reintroduce the bug.
    if config.bug_flags.get("bug_1", True):
        model.config.tie_word_embeddings = False
        model.config.decoder.tie_word_embeddings = False
    model.decoder.resize_token_embeddings(len(proc.tokenizer), mean_resizing=False)
    if config.bug_flags.get("bug_1", True):
        # Bug 10 (gate): weight-tie drift assert.  We clone unconditionally
        # when bug_1 fix is active so the optimizer tracks a distinct tensor.
        model.decoder.lm_head.weight = torch.nn.Parameter(
            model.decoder.lm_head.weight.data.clone(),
        )
    if config.bug_flags.get("bug_10", True):
        # Assertion: lm_head.weight must not share storage with the
        # embedding matrix after the clone (ablation-off leaves it tied).
        emb = model.decoder.get_input_embeddings().weight
        if model.decoder.lm_head.weight.data_ptr() == emb.data_ptr():
            from core.errors import TrainError
            raise TrainError("Bug 10 guard: lm_head still tied to embed_tokens.")
    # Bug 2 (gate): list-form convert_tokens_to_ids.  String-form returns
    # the ID of '<' and silently collapses F1 to 0; guard inverts for ablation.
    if config.bug_flags.get("bug_2", True):
        model.config.decoder_start_token_id = proc.tokenizer.convert_tokens_to_ids(
            ["<s_sroie>"],
        )[0]
    else:
        model.config.decoder_start_token_id = proc.tokenizer.convert_tokens_to_ids(
            "<s_sroie>",
        )
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.eos_token_id = proc.tokenizer.convert_tokens_to_ids(
        ["</s_sroie>"],
    )[0]
    model.config.vocab_size = model.config.decoder.vocab_size
    # Bug 9 (gate): Seq2SeqTrainer's predict_with_generate reads
    # generation_config snapshotted at from_pretrained.  Without mirroring
    # our SROIE-specific ids here, eval_f1 ≡ 0 while eval_loss drops normally.
    if config.bug_flags.get("bug_9", True):
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
    # Bug 11 (gate): transformers ≥4.48 adds num_items_in_batch to kwargs
    # which leaks into SwinModel.forward (no **kwargs) → TypeError.
    # Guard off keeps accepts_loss_kwargs at its default truthy value.
    if config.bug_flags.get("bug_11", True):
        model.accepts_loss_kwargs = False
    return proc, model


def _build_args(config: ExpConfig, out_dir: str) -> Seq2SeqTrainingArguments:
    cuda = torch.cuda.is_available()
    # Bug 4 (gate): precision selection bf16 vs fp16+grad_clip.  With the
    # guard active we prefer bf16 on Ampere+ (no NaN risk); guard-off
    # allows fp16 through without the max_grad_norm safety — i.e.
    # reintroduces the overflow-to-NaN failure mode.
    if config.bug_flags.get("bug_4", True):
        use_bf16 = (
            cuda and config.precision == "bf16" and torch.cuda.is_bf16_supported()
        )
        use_fp16 = cuda and config.precision == "fp16"
        effective_grad_norm = config.max_grad_norm
    else:
        use_bf16 = cuda and config.precision == "bf16"
        use_fp16 = cuda and config.precision == "fp16"
        effective_grad_norm = 0.0  # re-introduce the fp16 overflow path
    return Seq2SeqTrainingArguments(
        output_dir=out_dir, num_train_epochs=config.epochs_donut,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr, lr_scheduler_type=config.lr_scheduler_type,
        # Bug 13 (gate): HF Trainer uses warmup_steps when > 0, overriding
        # warmup_ratio.  With the guard active we force steps=0 when
        # warmup_ratio>0; guard-off intentionally re-introduces the override.
        warmup_ratio=config.warmup_ratio,
        warmup_steps=(
            0 if (config.warmup_ratio > 0
                  and config.bug_flags.get("bug_13", True))
            else config.warmup_steps
        ),
        weight_decay=config.weight_decay, bf16=use_bf16, fp16=use_fp16,
        max_grad_norm=effective_grad_norm,
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
    if _import_error is not None:
        raise ImportError(
            "torch and transformers are required for DONUT training. "
            "Run: pip install -r requirements.txt"
        ) from _import_error
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
    # P2 (RAG): when retrieval-augmented training is on, build the
    # Swin-CLS kNN bank once and swap the dataset constructor to the
    # neighbour-prefix variant; RAG-off path stays bit-identical.
    if config.rag_enabled:
        from models.donut_rag import _RAGSROIEDataset
        from models.retrieval_bank import build_bank

        bank = build_bank(data, config)
        train_ds: Any = _RAGSROIEDataset(data.train, proc, config, bank)
        val_ds: Any = _RAGSROIEDataset(data.val, proc, config, bank)
    else:
        train_ds = _SROIEDataset(data.train, proc, config)
        val_ds = _SROIEDataset(data.val, proc, config)
    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=val_ds,
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
    # Bug 9: load_best_model_at_end restored the best checkpoint in place,
    # including its stale generation_config (donut-base mBART defaults).
    # Re-pin BEFORE save_model; helper also asserts the round-trip on disk.
    start_id = proc.tokenizer.convert_tokens_to_ids(["<s_sroie>"])[0]
    eos_id = proc.tokenizer.convert_tokens_to_ids(["</s_sroie>"])[0]
    _persist_generation_config(
        model, out_dir, start_id, eos_id, proc.tokenizer.pad_token_id,
    )
    trainer.save_model(out_dir)
    proc.save_pretrained(out_dir)
    model.generation_config.save_pretrained(out_dir)  # belt-and-braces
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(config.output_dir, "donut_path.json"), "w") as f:
        json.dump({"model_path": out_dir}, f)
    return out_dir
