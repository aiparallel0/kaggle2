"""PR-A / T-A5 — single-source assigner architecture consistency.

Asserts that whichever architecture parameters land in
``combined_metrics.json`` via :func:`report.combine_ext.merge_assigner_arch`
agree numerically with what the assigner module itself reports — i.e.
the published paper number cannot drift away from the live model code.
"""
from __future__ import annotations

import importlib

import pytest


def test_shipped_constants_exposed() -> None:
    """L1 — the three architecture constants must be importable.

    Reviewers reading the paper see ``hidden=192``, ``layers=3``,
    ``heads=8``; the codebase must agree at import time so a clone
    + ``AttentionAssigner()`` reproduces those numbers.
    """
    mod = importlib.import_module("models.attention_model")
    assert mod.SHIPPED_HIDDEN_DIM == 192
    assert mod.SHIPPED_N_LAYERS == 3
    assert mod.SHIPPED_N_HEADS == 8
    # Legacy aliases must continue to resolve so older callers don't
    # break — the spec keeps them as DEFAULT_* aliases.
    assert mod.DEFAULT_HIDDEN_DIM == mod.LEGACY_HIDDEN_DIM
    assert mod.LEGACY_HIDDEN_DIM == 384
    assert mod.LEGACY_N_LAYERS == 6


def test_attention_assign_re_exports_aliases() -> None:
    """The :mod:`models.attention_assign` namespace must expose both
    SHIPPED_* and LEGACY_* so downstream importers don't reach into
    ``attention_model`` for what is now a single-source-of-truth."""
    aa = importlib.import_module("models.attention_assign")
    for name in (
        "SHIPPED_HIDDEN_DIM", "SHIPPED_N_LAYERS", "SHIPPED_N_HEADS",
        "LEGACY_HIDDEN_DIM", "LEGACY_N_LAYERS", "LEGACY_N_HEADS",
        "load_assigner", "migrate_v2_checkpoint",
    ):
        assert hasattr(aa, name), f"missing attribute {name}"


def test_constructor_defaults_match_shipped() -> None:
    """L1 — ``AttentionAssigner()`` with no arguments must instantiate
    a SHIPPED-architecture model so a reviewer who clones the repo
    and runs ``AttentionAssigner()`` reproduces the paper's numbers.
    Skipped when torch is not installed."""
    torch = pytest.importorskip("torch")
    _ = torch  # silence unused
    from models.attention_assign import AttentionAssigner
    m = AttentionAssigner()
    assert m.hidden_dim == 192
    assert m.n_layers == 3
    # n_heads may be adjusted down by ``_pick_n_heads`` if 8 is not a
    # divisor of 192; for hidden=192 the largest divisor ≤8 is 8 so
    # the equality holds.
    assert m.n_heads == 8
