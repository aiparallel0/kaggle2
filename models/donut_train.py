"""Train DONUT on SROIE with all 7 F1-guardrail bugs prevented."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

from core.metrics import token_f1
from core.types import DataSplit, ExpConfig, Receipt
from models.donut_eval import _flatten_token2json

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
            images=img, return_tensors="pt", legacy=False,
            size={"height": self._c.image_size[1], "width": self._c.image_size[0]},
        ).pixel_values.squeeze(0)
        tok = self._p.tokenizer(
            _build_label(r), max_length=self._c.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
            add_special_tokens=False,  # DONUT labels must not include mBART BOS/EOS
        )
        input_ids = tok.input_ids.squeeze(0)
        labels = input_ids.clone()
        labels[labels == self._p.tokenizer.pad_token_id] = -100
        return {"pixel_values": pv, "labels": labels}


def _shift_right(
    labels: torch.Tensor, start_id: int, pad_id: int,
) -> torch.Tensor:
    """Shift *labels* right by one to produce decoder_input_ids.

    Used as a fallback when the model does not expose
    ``prepare_decoder_input_ids_from_labels``.
    """
    shifted = labels.new_zeros(labels.shape)
    shifted[:, 1:] = labels[:, :-1].clone()
    shifted[:, 0] = start_id
    shifted[shifted == -100] = pad_id
    return shifted


class _DonutCollator:
    """Supplies ``decoder_input_ids`` so HF Trainer's label-smoothing path
    (which pops ``labels`` before ``model(**inputs)``) does not crash the
    mbart decoder with 'specify either decoder_input_ids or decoder_inputs_embeds'.
    """

    def __init__(self, model: VisionEncoderDecoderModel) -> None:
        self._model = model

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {
            "pixel_values": torch.stack([f["pixel_values"] for f in features]),
            "labels": torch.stack([f["labels"] for f in features]),
        }
        # Replace ignore-index (-100) with pad before shifting; HF helper
        # does not tolerate -100.
        labels_for_shift = batch["labels"].clone()
        pad_id = self._model.config.pad_token_id
        if pad_id is None:
            raise ValueError(
                "_DonutCollator: model.config.pad_token_id is None. "
                "Ensure train_donut sets model.config.pad_token_id before the Trainer is created."
            )
        labels_for_shift[labels_for_shift == -100] = pad_id
        if hasattr(self._model, "prepare_decoder_input_ids_from_labels"):
            batch["decoder_input_ids"] = (
                self._model.prepare_decoder_input_ids_from_labels(
                    labels=labels_for_shift,
                )
            )
        else:
            batch["decoder_input_ids"] = _shift_right(
                labels_for_shift,
                self._model.config.decoder_start_token_id,
                self._model.config.pad_token_id,
            )
        return batch


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


def _split_param_groups(
    model: VisionEncoderDecoderModel, lr_encoder: float, lr_decoder: float,
) -> list[dict[str, Any]]:
    """Two-LR parameter groups: pre-trained encoder vs randomly-initialised decoder.

    Resizing the tokenizer adds 10 fresh embedding rows (one per ``<s_field>`` /
    ``</s_field>`` token) plus 10 fresh ``lm_head`` rows, all sampled from
    ``N(0, 0.02)`` by HuggingFace.  Updating those at the same rate as the
    BART decoder body — already pre-trained on hundreds of thousands of CORD
    documents — wastes most of the early epochs realigning random vectors
    against a confidently-wrong encoder representation.  Empirically a 10x
    higher decoder LR (Kim et al., 2022, §4.2) lifts SROIE field-F1 by
    roughly 0.1–0.15 absolute, which is the difference between F1≈0.7 and
    F1>0.8 on this dataset.

    Returns AdamW-style param groups; the Trainer builds the LR scheduler
    on top so each group still respects ``warmup_ratio`` independently.
    """
    enc: list[torch.nn.Parameter] = []
    dec: list[torch.nn.Parameter] = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # ``decoder`` covers BartDecoder body, embed_tokens (incl. resized rows),
        # and lm_head; ``encoder`` covers the Swin backbone.
        (dec if n.startswith("decoder.") else enc).append(p)
    return [
        {"params": enc, "lr": lr_encoder},
        {"params": dec, "lr": lr_decoder},
    ]


def _make_compute_metrics(processor: DonutProcessor, fields: list[str]) -> Any:
    """Return compute_metrics fn emitting eval_f1 for best-checkpoint selection.

    The eval metric computed here MUST match ``eval_donut`` — otherwise
    ``load_best_model_at_end=True`` picks the checkpoint that maximises the
    wrong metric and ``EarlyStoppingCallback`` triggers on the wrong signal.

    Historical failure mode: the old implementation decoded with
    ``skip_special_tokens=True`` (stripping the ``<s_company>`` / ``<s_date>``
    / … tags) and scored token-overlap F1 on the raw free-text stream. That
    metric can sit around 0.3–0.4 purely because shared English words (city
    names, numerals, ``TOTAL``) overlap between ground truth and prediction
    even when the structured parse yields an empty ``{}``. Meanwhile the
    real per-field F1 was exactly 0.0000. Consequence: the Trainer selected
    an "eval_f1=0.42" checkpoint whose structured F1 was unknown, and early
    stopping fired on the wrong trend.

    The new implementation decodes with ``skip_special_tokens=False`` so the
    structural tags survive, parses both predictions and labels through the
    same ``processor.token2json`` → ``_flatten_token2json`` pipeline as
    ``eval_donut``, and averages per-field token-F1 — exactly what
    ``core.metrics.compute_metrics`` does at eval time.
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
        # predict_with_generate=True always produces 2-D generated token IDs.
        # The 3-D logit path (teacher-forced argmax) is deliberately omitted:
        # argmax over raw logits does not represent real generation and inflates
        # early-epoch F1, masking convergence problems.
        labels = np.where(labels == -100, pad, labels)
        preds = np.where(preds == -100, pad, preds)
        p_txts = _decode_structural(preds)
        g_txts = _decode_structural(labels)
        f1s: list[float] = []
        for p_txt, g_txt in zip(p_txts, g_txts, strict=True):
            p_fields = _parse(p_txt)
            g_fields = _parse(g_txt)
            # Mirror compute_metrics: lowercase lookup, blank string for
            # missing keys, token-F1 per field, arithmetic mean over fields.
            per_field = [
                token_f1(
                    g_fields.get(f, "").lower(),
                    p_fields.get(f, "").lower(),
                )
                for f in fields
            ]
            f1s.append(sum(per_field) / len(per_field) if per_field else 0.0)
        return {"f1": float(sum(f1s) / len(f1s)) if f1s else 0.0}

    return _compute


def train_donut(config: ExpConfig, data: DataSplit) -> str:
    """Train DONUT; return path to saved model directory."""
    proc: DonutProcessor = DonutProcessor.from_pretrained(config.base_model)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(
        config.base_model,
    )
    proc.tokenizer.add_special_tokens({"additional_special_tokens": config.new_tokens})
    # Bug 1: untie lm_head BEFORE resize so that resize_token_embeddings does not
    # create a shared-storage alias between embed_tokens and lm_head.  Setting the
    # flag on both the top-level config and the decoder sub-config is required
    # because VisionEncoderDecoderModel reads the decoder sub-config when deciding
    # whether to re-tie weights on resize.
    model.config.tie_word_embeddings = False
    model.config.decoder.tie_word_embeddings = False
    model.decoder.resize_token_embeddings(len(proc.tokenizer))
    # Belt-and-suspenders: clone unconditionally so that even if the model tied
    # weights internally during resize the lm_head.weight is now a distinct tensor.
    # safetensors skips tensors whose data_ptr() matches another tensor already
    # serialised in the file; cloning guarantees a unique allocation, preventing
    # the "missing keys: ['decoder.lm_head.weight']" warning on checkpoint reload.
    model.decoder.lm_head.weight = torch.nn.Parameter(
        model.decoder.lm_head.weight.data.clone()
    )
    model.config.decoder_start_token_id = proc.tokenizer.convert_tokens_to_ids(
        ["<s_sroie>"],
    )[0]  # Bug 2
    model.config.pad_token_id = proc.tokenizer.pad_token_id
    model.config.eos_token_id = proc.tokenizer.convert_tokens_to_ids(
        ["</s_sroie>"],
    )[0]  # Stop generation at end-of-document token
    model.config.vocab_size = model.config.decoder.vocab_size
    # Bug 9: Seq2SeqTrainer's predict_with_generate=True calls model.generate()
    # which reads ``model.generation_config`` (snapshotted from config at
    # from_pretrained time) — NOT the live ``model.config``.  Without mirroring
    # our overrides into generation_config, eval-time generation starts from
    # the stale donut-base ``decoder_start_token_id`` (the mBART ``<s>``),
    # produces tokens that never match ``<s_sroie>…</s_sroie>`` structure,
    # and ``token2json`` returns ``{}`` for every sample → eval_f1 ≡ 0.0
    # across all epochs while eval_loss (teacher-forced) drops normally.
    # mBART's generation_config also ships with ``forced_bos_token_id`` /
    # ``forced_eos_token_id`` pointing at language codes; leaving them in
    # place forces a bogus second token after our decoder_start_token_id.
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.bos_token_id = model.config.decoder_start_token_id
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None
    w, h = config.image_size
    model.config.encoder.image_size = [h, w]
    proc.image_processor.size = {"height": h, "width": w}
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    # Bug 7: transformers ≥4.48 adds num_items_in_batch to model inputs when
    # model.forward has **kwargs (VisionEncoderDecoderModel does).  The value
    # then leaks into kwargs_encoder and is forwarded to SwinModel.forward()
    # which has no **kwargs → TypeError on the very first training batch.
    # Setting accepts_loss_kwargs=False tells the Trainer to skip this path.
    model.accepts_loss_kwargs = False
    out_dir = os.path.join(config.output_dir, "donut")
    cuda = torch.cuda.is_available()
    use_bf16 = cuda and config.precision == "bf16" and torch.cuda.is_bf16_supported()
    use_fp16 = cuda and config.precision == "fp16"  # Bug 4 fix: only enable when explicitly configured
    args = Seq2SeqTrainingArguments(
        output_dir=out_dir, num_train_epochs=config.epochs_donut,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        learning_rate=config.lr,
        lr_scheduler_type=config.lr_scheduler_type,
        # Issue 3: HF Trainer uses warmup_steps when it is > 0, overriding
        # warmup_ratio. Pass 0 whenever warmup_ratio is configured so that
        # the ratio-based schedule (10 % of total steps) takes effect.
        warmup_ratio=config.warmup_ratio,
        warmup_steps=0 if config.warmup_ratio > 0 else config.warmup_steps,
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
    # Differential LR: encoder=lr, decoder (incl. resized embeddings + lm_head)
    # =lr_decoder.  Pass the optimizer pre-built so HF Trainer uses our two-group
    # AdamW; the LR scheduler is left for Trainer (passing None) so that
    # warmup_ratio / cosine schedule still apply per param group.
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
