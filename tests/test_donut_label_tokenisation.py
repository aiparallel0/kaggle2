"""Regression tests for DONUT label-tokenisation correctness.

Three bugs guarded here:

1. add_special_tokens=False — mBART's tokenizer adds BOS (<s>, id 0) and EOS
   (</s>, id 2) when add_special_tokens is left at its default True.  DONUT
   labels must NOT include those sentinel tokens because they corrupt the
   decoder_input_ids shift: the model learns to predict BOS as the first
   generated token, wasting a position and misaligning all subsequent
   predictions.

2. eos_token_id must be set to </s_sroie> during training — without it the
   model generates padding tokens all the way to max_length (768) after the
   structured output is complete, wasting compute and sometimes confusing
   token2json.

3. eos_token_id must be resolved and passed to model.generate() during eval —
   so early_stopping=True actually triggers at the end-of-document token rather
   than running to max_length on every receipt.

All three bugs are verifiable at source level without a GPU / HuggingFace install.
"""
from __future__ import annotations

import inspect

from models import donut_eval, donut_train


def test_tokenizer_uses_add_special_tokens_false() -> None:
    """_SROIEDataset.__getitem__ must pass add_special_tokens=False."""
    src = inspect.getsource(donut_train._SROIEDataset.__getitem__)  # type: ignore[attr-defined]
    assert "add_special_tokens=False" in src, (
        "_SROIEDataset.__getitem__ must tokenize labels with add_special_tokens=False. "
        "Without it mBART prepends <s> (BOS, id=0) and appends </s> (EOS, id=2) to "
        "every label, shifting all DONUT field tokens right by one position and causing "
        "the model to predict BOS as its first output token."
    )


def test_train_donut_sets_eos_token_id() -> None:
    """train_donut must configure model.config.eos_token_id to </s_sroie>."""
    src = inspect.getsource(donut_train.train_donut)
    assert "eos_token_id" in src, (
        "train_donut must set model.config.eos_token_id to the </s_sroie> token ID. "
        "Without this the model generates until max_length (768 tokens) after "
        "producing the complete structured output, wasting compute and risking "
        "token2json parse failures from trailing pad tokens."
    )


def test_eval_donut_resolves_eos_token_id() -> None:
    """eval_donut must resolve eos_token_id and pass it to model.generate()."""
    src = inspect.getsource(donut_eval.eval_donut)
    assert "eos_token_id" in src, (
        "eval_donut must resolve eos_token_id (</s_sroie>) and pass it to "
        "model.generate() so that early_stopping=True actually triggers when "
        "the end-of-document token is produced."
    )
