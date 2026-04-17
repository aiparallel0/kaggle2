"""Regression test: train_donut / train_trocr must set accepts_loss_kwargs=False.

In transformers >=4.48 the Trainer inspects **kwargs in model.forward to decide
whether to pass num_items_in_batch into the model inputs.
VisionEncoderDecoderModel.forward has **kwargs, so without the explicit flag
the Trainer would merge num_items_in_batch into inputs, which then leaks into
kwargs_encoder and crashes SwinModel / ViT (those have no **kwargs).

This test verifies the guard is present at source level, which works without
requiring the full GPU/torch stack.
"""
from __future__ import annotations

import inspect

from models import donut_train, trocr_train


def test_train_donut_sets_accepts_loss_kwargs_false() -> None:
    src = inspect.getsource(donut_train.train_donut)
    assert "accepts_loss_kwargs" in src, (
        "train_donut must set model.accepts_loss_kwargs=False before Trainer init "
        "(Bug 7: num_items_in_batch crashes SwinModel in transformers >=4.48)"
    )


def test_train_trocr_sets_accepts_loss_kwargs_false() -> None:
    src = inspect.getsource(trocr_train.train_trocr)
    assert "accepts_loss_kwargs" in src, (
        "train_trocr must set model.accepts_loss_kwargs=False before Trainer init "
        "(Bug 7: num_items_in_batch crashes ViT encoder in transformers >=4.48)"
    )
