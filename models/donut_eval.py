"""Evaluate DONUT on SROIE test split with all Bug 2/3/8/9 guards.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: runs DONUT inference with decoder_start_token_id=<s_sroie> (Bug 2),
    flattens token2json output (Bug 3/8), disables forced EOS (Bug 9),
    normalizes TOTAL values for symmetric metric comparison, and persists
    per-receipt diagnostics to donut_eval_diag.json so forensics survive
    truncated operator logs (see _write_eval_diag).
"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from core.errors import EvalError
from core.metrics import compute_metrics
from core.types import EvalBundle, ExpConfig, Field, Metrics, Prediction, Receipt
from models.donut_parse import _flatten_token2json
from models.donut_parse import token2json_safe as _token2json_safe

_log = logging.getLogger("kaggle2")

_import_error: ImportError | None = None
try:
    import torch
    from transformers import DonutProcessor, VisionEncoderDecoderModel
except ImportError as _exc:  # lightweight CI — torch/transformers not installed
    _import_error = _exc

__all__ = ["eval_donut", "_token2json_safe", "normalize_total"]

_CURRENCY_RE = re.compile(r"(?:\brm|\busd|\bmyr)\.?\s*", re.IGNORECASE)
_CURRENCY_SYMS = ("$", "₹", "€", "£")
_THOUSANDS_RE = re.compile(r"^\d{1,3}(,\d{3})+(\.\d{1,2})?$")
_NUMERIC_TOKEN_RE = re.compile(r"^\d+(\.\d{1,2})?$")


def normalize_total(s: str) -> str:
    """Normalize TOTAL string for symmetric DONUT/GT comparison."""
    if not s:
        return s
    t = _CURRENCY_RE.sub("", s)
    for sym in _CURRENCY_SYMS:
        t = t.replace(sym, "")
    t = t.strip()
    if _THOUSANDS_RE.match(t):
        t = t.replace(",", "")
    tokens = [tok for tok in t.split() if _NUMERIC_TOKEN_RE.match(tok)]
    if tokens:
        t = tokens[-1]
    try:
        return f"{float(t):.2f}"
    except ValueError:
        return t


def _apply_total_normalizer(preds: list[Prediction], gts: list[Receipt]) -> tuple[
    list[Prediction], list[Receipt],
]:
    """Symmetric per-field normalisation (preds + GT) so token-F1 measures
    semantic match.  Uses the DONUT-specific normaliser map (legacy
    ``normalize_address``, no FOCUS punctuation pass) so headline F1 is
    bit-identical to PR #110.  Routes through the shared helper so the
    *exact same* ``(preds', receipts')`` pair flows out to the extended-
    metrics producer in :mod:`stages.eval_producers` (PR #110 follow-up:
    fixes ``F1 > max(P, R)`` in ``extended_metrics.json``)."""
    # Lazy import — keeps DONUT eval importable in torch-free CI.
    from models.normalize_bundle import FIELD_NORMALISERS_DONUT, normalize_bundle
    return normalize_bundle(preds, gts, FIELD_NORMALISERS_DONUT)


def _load(model_path: str) -> tuple[Any, Any, str]:
    if not Path(model_path).exists():
        raise EvalError(f"DONUT model not found at {model_path}")
    processor: DonutProcessor = DonutProcessor.from_pretrained(model_path)
    model: VisionEncoderDecoderModel = VisionEncoderDecoderModel.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    # Bug 9 (eval side): clear donut-base's forced_*_token_id so generation
    # can emit </s_sroie> instead of mBART's </s> (id 2).
    model.generation_config.forced_bos_token_id = None
    model.generation_config.forced_eos_token_id = None
    return processor, model, device


def _resolve_start_eos(processor: Any, model: Any) -> tuple[int, int | None]:
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
    return start_id, eos_id


def _write_eval_diag(
    processor: Any,
    model: Any,
    start_id: int,
    eos_id: int | None,
    samples: list[dict[str, Any]],
    out_dir: str,
) -> None:
    """Write donut_eval_diag.json with model/tokenizer metadata + 5 sample receipts.

    Persists lm_head_out_features (Bug 1 indicator), both token ids (Bug 2),
    raw token2json output (Bug 3/8), and per-receipt parsed vs. GT comparison
    so forensics are available even when the operator log is truncated.
    """
    tok = processor.tokenizer
    lm = model.decoder.lm_head
    gc = model.generation_config
    s_id = tok.convert_tokens_to_ids(["<s_sroie>"])[0]
    diag: dict[str, Any] = {
        "image_processor_size": processor.image_processor.size,
        "decoder_start_token_id": start_id,
        "eos_token_id": eos_id,
        "pad_token_id": model.config.pad_token_id,
        "tokenizer_has_s_sroie": s_id != tok.unk_token_id,
        "tokenizer_s_sroie_id": s_id,
        "tokenizer_eos_s_sroie_id": tok.convert_tokens_to_ids(["</s_sroie>"])[0],
        "model_config_vocab_size": model.config.decoder.vocab_size,
        "tokenizer_vocab_size": len(tok),
        "lm_head_out_features": lm.weight.shape[0],
        "generation_config": {
            "forced_bos_token_id": gc.forced_bos_token_id,
            "forced_eos_token_id": gc.forced_eos_token_id,
            "decoder_start_token_id": gc.decoder_start_token_id,
            "eos_token_id": gc.eos_token_id,
            "bos_token_id": getattr(gc, "bos_token_id", None),
        },
        "samples": samples,
    }
    out_path = os.path.join(out_dir, "donut_eval_diag.json")
    with open(out_path, "w") as fh:
        json.dump(diag, fh, indent=2, default=str)
    _log.info("donut_eval_diag written → %s", out_path)


def eval_donut(
    config: ExpConfig, test: list[Receipt],
) -> tuple[Metrics, list[Prediction], list[Receipt]]:
    """Run DONUT inference; return ``(Metrics, normalised preds, normalised gold)``.

    The third return value (the field-normalised gold receipts) is
    surfaced so the extended-metrics producer in
    :mod:`stages.eval_producers` can build its ``EvalBundle`` from
    the *same* ``(preds, receipts)`` pair the headline F1 scorer
    saw.  Before PR #111 the producer received raw ``data.test``
    while the preds were normalised → per-field precision / recall /
    bootstrap CI collapsed and ``F1 > max(P, R)`` appeared in
    ``extended_metrics.json``.
    """
    if _import_error is not None:
        raise ImportError(
            "torch and transformers are required for DONUT evaluation. "
            "Run: pip install -r requirements.txt"
        ) from _import_error
    model_path = os.path.join(config.output_dir, "donut")
    processor, model, device = _load(model_path)
    start_id, eos_id = _resolve_start_eos(processor, model)
    num_beams = config.num_beams
    max_len = config.max_length
    # Pin inference image size on the image_processor (mirrors donut_train);
    # avoids transformers 4.48's per-call size= kwarg misrouting (PR #87).
    processor.image_processor.size = {
        "height": config.image_size[1], "width": config.image_size[0],
    }
    predictions: list[Prediction] = []
    diag_samples: list[dict[str, Any]] = []
    from PIL import Image

    # P2 (RAG, inference mirror): build (or no-op) a bank and use it to
    # prefix decoder_input_ids with <retrieved>...</retrieved> tokens.
    # When rag_enabled is False :func:`build_rag_prompt` returns just
    # [start_id], so the RAG-off path is bit-identical to pre-P2.
    from models.donut_rag import build_rag_prompt
    from models.retrieval_bank import build_bank, empty_bank

    if config.rag_enabled:
        from data.sroie import download_sroie, load_or_create_split
        data_path = download_sroie(config)
        split = load_or_create_split(config, data_path)
        rag_bank = build_bank(split, config)
    else:
        rag_bank = empty_bank()
    with torch.no_grad():
        for i, rec in enumerate(test):
            img = Image.open(rec.image_path).convert("RGB")
            pv = processor(
                images=img, return_tensors="pt", legacy=False,
            ).pixel_values.to(device)
            prompt_ids = build_rag_prompt(
                rag_bank, (str(rec.image_path), config, processor.tokenizer),
            )
            if len(prompt_ids) > 1:
                # RAG prefix present — pass via decoder_input_ids so HF
                # generate() seeds beam search with the neighbour context.
                dec_ids = torch.tensor([prompt_ids], device=device)
                out = model.generate(
                    pv, decoder_input_ids=dec_ids, eos_token_id=eos_id,
                    max_length=max_len, num_beams=num_beams, early_stopping=True,
                )
            else:
                out = model.generate(
                    pv, decoder_start_token_id=start_id, eos_token_id=eos_id,
                    max_length=max_len, num_beams=num_beams, early_stopping=True,
                )
            tokens = processor.batch_decode(out, skip_special_tokens=False)[0]
            # Bug 3/12 gates: _token2json_safe handles list→dict merge and
            # outer <s_sroie> wrapper flattening; degrades to {} when off.
            if (
                config.bug_flags.get("bug_3", True)
                and config.bug_flags.get("bug_12", True)
            ):
                parsed = _token2json_safe(processor, tokens)
            elif config.bug_flags.get("bug_3", True):
                # Only Bug 3 fix active — flatten but keep outer wrapper.
                raw = processor.token2json(tokens)
                parsed = _flatten_token2json(raw if isinstance(raw, list) else [raw])
            else:
                raw = processor.token2json(tokens)
                parsed = raw if isinstance(raw, dict) else {}
            if i == 0:
                # One-shot diagnostic: catches preprocessor pin loss, Bug 2
                # regression (start/eos ids), lm_head/gen-config corruption
                # (raw tokens), and Bug 3/8/12 parser regressions (parsed).
                _log.info(
                    "donut_eval diag: image_size=%s start_id=%d eos_id=%s "
                    "tokens[:200]=%r parsed_keys=%s",
                    processor.image_processor.size,
                    int(model.config.decoder_start_token_id),
                    model.config.eos_token_id,
                    tokens[:200],
                    sorted(parsed.keys()),
                )
            if i < 5:
                diag_samples.append({
                    "image_id": rec.image_path.stem,
                    "tokens_full": tokens,
                    "raw_token2json": processor.token2json(tokens),
                    "parsed": parsed,
                    "gt": {f.name.lower(): f.value for f in rec.fields},
                })
            fields = [Field(name=k, value=v) for k, v in parsed.items()]
            predictions.append(Prediction(receipt_id=rec.image_path.stem, fields=fields))
    # Symmetric numeric normalization for the TOTAL field (DONUT only):
    # matches the paper's "normalized numeric comparison for the Total field"
    # so F1/EM/NED measure semantic match rather than incidental formatting.
    out_dir = model_path
    _write_eval_diag(processor, model, start_id, eos_id, diag_samples, out_dir)
    norm_preds, norm_test = _apply_total_normalizer(predictions, test)
    metrics = compute_metrics(EvalBundle(
        predictions=norm_preds, receipts=norm_test, fields=config.fields,
    ))
    with open(os.path.join(out_dir, "donut_metrics.json"), "w") as f:
        json.dump(
            {
                "global_f1": metrics.global_f1, "global_ned": metrics.global_ned,
                "global_em": metrics.global_em, "per_field_f1": metrics.per_field_f1,
            }, f, indent=2,
        )
    return metrics, norm_preds, norm_test
