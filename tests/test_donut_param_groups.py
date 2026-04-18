"""Verify that train_donut applies the differential decoder learning rate.

The 10 special tokens added by ``proc.tokenizer.add_special_tokens`` start
from ``N(0, 0.02)``; when both encoder and decoder share ``lr=5e-5`` those
fresh embedding rows take 8–10 epochs to align with the encoder
representation, leaving ~0.10 absolute F1 on the table.  Configuring
``lr_decoder`` (default 1e-4 = 2× ``lr``, but practitioners use up to 10×)
fixes this without changing the encoder, but only if the optimizer actually
gets the two-group parameter list.

These tests:

1. Source-level: ``train_donut`` constructs a two-group AdamW.
2. Functional: ``_split_param_groups`` puts every decoder.* tensor in the
   second group at ``lr_decoder`` and every other tensor in the first at
   ``lr``.
"""
from __future__ import annotations

import inspect
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from models import donut_train  # noqa: E402


def test_train_donut_uses_two_group_optimizer() -> None:
    src = inspect.getsource(donut_train.train_donut)
    assert "_split_param_groups" in src, (
        "train_donut must call _split_param_groups so the resized decoder "
        "embeddings + lm_head get lr_decoder, not the encoder's lr."
    )
    assert "optimizers=(optimizer, None)" in src, (
        "train_donut must hand the pre-built two-group optimizer to the "
        "Trainer (Trainer still builds the LR scheduler from args)."
    )


def test_split_param_groups_routes_decoder_to_higher_lr() -> None:
    """A toy module with ``decoder.X`` and ``encoder.X`` parameters must
    end up in distinct groups with distinct LRs."""

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)

    m: Any = _Tiny()
    groups = donut_train._split_param_groups(m, lr_encoder=5e-5, lr_decoder=1e-4)
    assert len(groups) == 2
    enc_group, dec_group = groups
    assert enc_group["lr"] == 5e-5
    assert dec_group["lr"] == 1e-4
    # decoder group must contain at least the two decoder linear params (W, b)
    assert len(dec_group["params"]) >= 2
    assert len(enc_group["params"]) >= 2
    # No parameter should appear in both groups (would double-step it).
    enc_ids = {id(p) for p in enc_group["params"]}
    dec_ids = {id(p) for p in dec_group["params"]}
    assert not (enc_ids & dec_ids), "param leak: tensor present in both groups"


def test_split_param_groups_skips_frozen_params() -> None:
    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Linear(2, 2)
            self.decoder = torch.nn.Linear(2, 2)
            for p in self.encoder.parameters():
                p.requires_grad = False

    m: Any = _Tiny()
    groups = donut_train._split_param_groups(m, lr_encoder=5e-5, lr_decoder=1e-4)
    enc_group, _ = groups
    assert enc_group["params"] == [], (
        "Frozen parameters must not appear in any param group "
        "(AdamW would still consume optimizer state for them otherwise)."
    )
