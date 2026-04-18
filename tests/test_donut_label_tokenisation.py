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


def test_in_training_metric_matches_eval_metric() -> None:
    """``load_best_model_at_end`` must select the checkpoint that maximises
    the same metric ``eval_donut`` reports — otherwise the Trainer picks a
    checkpoint good at raw token-overlap but bad at structured per-field F1.

    Source-level guards on the in-training ``compute_metrics`` factory:

    * Decodes with ``skip_special_tokens=False`` so the ``<s_field>`` /
      ``</s_field>`` tags survive and ``token2json`` can parse them.
    * Parses through the same ``processor.token2json`` + flatten pipeline
      as ``eval_donut``.
    * Takes ``fields`` from the experiment config and averages per-field
      token-F1, matching ``core.metrics.compute_metrics``.
    """
    src = inspect.getsource(donut_train._make_compute_metrics)  # type: ignore[attr-defined]
    assert "skip_special_tokens=False" in src, (
        "_make_compute_metrics must decode with skip_special_tokens=False — "
        "the structural tags must survive decoding so token2json can parse "
        "the per-field dict. Stripping them reduces the metric to raw "
        "token-overlap F1 on free text, which diverges from eval_donut."
    )
    assert "token2json" in src, (
        "_make_compute_metrics must parse predictions+labels through "
        "processor.token2json (plus the shared _flatten_token2json wrapper "
        "fix) so the in-training metric matches eval_donut's per-field F1. "
        "Without this, load_best_model_at_end picks the wrong checkpoint."
    )
    assert "_flatten_token2json" in src, (
        "_make_compute_metrics must apply _flatten_token2json after "
        "token2json so the outer <s_sroie> wrapper is unwrapped exactly as "
        "eval_donut does it. Any divergence here recreates the 'eval_f1 in "
        "training was 0.34 but post-training F1 was 0.00' failure mode."
    )
    # Signature sanity: must now accept the fields list so the metric can
    # iterate over the exact same field set as eval_donut + compute_metrics.
    sig_src = inspect.getsource(donut_train._make_compute_metrics)  # type: ignore[attr-defined]
    assert "fields" in sig_src.splitlines()[0], (
        "_make_compute_metrics(processor, fields) signature is required so "
        "the training metric iterates the configured field list."
    )
