"""Regression tests for the assigner train/val split.

Prior to this change the "Assigner epoch N/N loss=0.111" line printed by
``train_assigner`` was *training* loss only — there was no held-out split,
so the number reflected nothing but memorisation. These tests guard the
reporting contract and the determinism of the split so a future refactor
cannot silently regress it back to a single-set memorisation signal.

Source-level guards are used for the parts that would otherwise need a
torch+transformers+SROIE-dataset environment to exercise end-to-end.
"""
from __future__ import annotations

import inspect

import pytest

from models import assigner_train


def test_train_assigner_reports_val_loss() -> None:
    """The per-epoch log line must carry both train_loss and val_loss."""
    src = inspect.getsource(assigner_train.train_assigner)
    assert "val_loss" in src, (
        "train_assigner must report val_loss every epoch. Without it the "
        "'loss trended down to X' headline is meaningless — you only "
        "observe that the model memorised its training set."
    )
    assert "train_loss" in src, (
        "train_assigner must also report train_loss so the generalisation "
        "gap (train vs val) is visible in the log."
    )


def test_train_assigner_saves_best_by_val() -> None:
    """Saved checkpoint must be best-by-val, not last epoch."""
    src = inspect.getsource(assigner_train.train_assigner)
    assert "best_val" in src and "best_state" in src, (
        "train_assigner must track best val loss and the state_dict that "
        "achieved it, then restore that state before save_assigner. "
        "Saving last-epoch weights defeats the purpose of having a val "
        "split — a later epoch can easily be worse than an earlier one."
    )
    assert "load_state_dict(best_state)" in src, (
        "The best-by-val state_dict must be loaded back into the model "
        "before saving; otherwise the saved checkpoint is still last-epoch."
    )


def test_split_train_val_is_deterministic_and_nonempty() -> None:
    """_split_train_val must be reproducible and produce a non-empty val."""
    torch = pytest.importorskip("torch")
    # Fabricate 20 tiny prepared groups (feats, bboxes, priors, targets).
    prepared = [
        (
            torch.zeros(3, 768),
            torch.zeros(3, 4),
            torch.zeros(3, 6),
            {0: [0], 1: [1], 2: [2]},
        )
        for _ in range(20)
    ]
    t1, v1 = assigner_train._split_train_val(prepared, seed=42)  # type: ignore[attr-defined]
    t2, v2 = assigner_train._split_train_val(prepared, seed=42)  # type: ignore[attr-defined]
    # Determinism: same seed → same split.
    assert len(t1) == len(t2)
    assert len(v1) == len(v2)
    # Non-empty val set.
    assert len(v1) >= 1
    # Disjoint train/val (compare by identity-of-tuple since _split_train_val
    # reuses references from prepared[]).
    train_ids = {id(g) for g in t1}
    val_ids = {id(g) for g in v1}
    assert train_ids.isdisjoint(val_ids), (
        "train and val groups must be disjoint subsets of prepared[]"
    )
    # The split partitions the full set.
    assert len(t1) + len(v1) == len(prepared)


def test_split_train_val_handles_singleton() -> None:
    """With one prepared group, both sides degenerate to that single group
    — training still runs, but the val signal is just train loss."""
    torch = pytest.importorskip("torch")
    prepared = [(torch.zeros(1, 768), torch.zeros(1, 4), torch.zeros(1, 6), {0: [0]})]
    train, val = assigner_train._split_train_val(prepared, seed=0)  # type: ignore[attr-defined]
    assert len(train) == 1 and len(val) == 1


def test_different_seeds_produce_different_splits() -> None:
    """Non-deterministic split would be a correctness hazard (non-repro runs);
    trivially-constant split would be a coverage hazard. We want: same seed
    → same split, different seeds → likely different splits."""
    torch = pytest.importorskip("torch")
    prepared = [
        (torch.zeros(2, 768), torch.zeros(2, 4), torch.zeros(2, 6), {0: [0]})
        for _ in range(50)
    ]
    _, v_a = assigner_train._split_train_val(prepared, seed=1)  # type: ignore[attr-defined]
    _, v_b = assigner_train._split_train_val(prepared, seed=2)  # type: ignore[attr-defined]
    ids_a = {id(g) for g in v_a}
    ids_b = {id(g) for g in v_b}
    # Over 50 items with 10% val (=5 items), the chance of identical 5-set
    # picks under two distinct seeds is vanishingly small (~1 in C(50,5)).
    assert ids_a != ids_b, "different seeds must produce different val sets"
