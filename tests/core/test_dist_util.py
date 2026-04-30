"""Unit tests for core.dist_util.is_rank_zero env-var-first detection.

Regression test for the 8× RTX 5090 swarm crash: under torchrun + HF
Trainer + accelerate, ``torch.distributed.get_rank()`` was observed
returning 0 on every rank after ``Trainer.train()`` returned, bypassing
the rank-0 gate on cleanup writes and racing
``generation_config.json``.  These tests pin the contract that env vars
(set authoritatively by torchrun) win over a degraded process-group
state.
"""
from __future__ import annotations

from core.dist_util import is_rank_zero


def test_is_rank_zero_default_true_when_no_env(monkeypatch: object) -> None:
    """Single-process / non-DDP: returns True so legacy paths behave."""
    for var in ("RANK", "LOCAL_RANK"):
        monkeypatch.delenv(var, raising=False)  # type: ignore[attr-defined]
    assert is_rank_zero() is True


def test_is_rank_zero_true_when_rank_env_zero(monkeypatch: object) -> None:
    monkeypatch.setenv("RANK", "0")  # type: ignore[attr-defined]
    monkeypatch.setenv("LOCAL_RANK", "0")  # type: ignore[attr-defined]
    assert is_rank_zero() is True


def test_is_rank_zero_false_when_rank_env_nonzero(
    monkeypatch: object,
) -> None:
    """Critical regression: rank 2 worker must be detected as non-zero
    even if torch.distributed reports get_rank()==0 (the failure mode
    observed in the 8× RTX 5090 swarm crash)."""
    monkeypatch.setenv("RANK", "2")  # type: ignore[attr-defined]
    monkeypatch.setenv("LOCAL_RANK", "2")  # type: ignore[attr-defined]
    assert is_rank_zero() is False


def test_is_rank_zero_falls_back_to_local_rank(monkeypatch: object) -> None:
    """When RANK is unset/garbage, LOCAL_RANK is the next signal."""
    monkeypatch.delenv("RANK", raising=False)  # type: ignore[attr-defined]
    monkeypatch.setenv("LOCAL_RANK", "3")  # type: ignore[attr-defined]
    assert is_rank_zero() is False


def test_is_rank_zero_skips_empty_env(monkeypatch: object) -> None:
    """Empty string env vars must not crash with ValueError; we skip
    to the next signal so a misconfigured launcher still gets a
    correct answer rather than a hard crash mid-cleanup."""
    monkeypatch.setenv("RANK", "")  # type: ignore[attr-defined]
    monkeypatch.setenv("LOCAL_RANK", "1")  # type: ignore[attr-defined]
    assert is_rank_zero() is False


def test_is_rank_zero_env_beats_dist(monkeypatch: object) -> None:
    """Env var trumps a (mocked) distributed process group reporting
    rank 0.  This is the core regression: accelerate post-train
    teardown left the default process group reporting rank 0 on every
    worker."""
    import sys
    import types

    fake_dist = types.ModuleType("torch.distributed")
    fake_dist.is_available = lambda: True  # type: ignore[attr-defined]
    fake_dist.is_initialized = lambda: True  # type: ignore[attr-defined]
    fake_dist.get_rank = lambda: 0  # type: ignore[attr-defined]
    fake_torch = sys.modules.get("torch")
    if fake_torch is None:
        fake_torch = types.ModuleType("torch")
        monkeypatch.setitem(sys.modules, "torch", fake_torch)  # type: ignore[attr-defined]
    monkeypatch.setitem(  # type: ignore[attr-defined]
        sys.modules, "torch.distributed", fake_dist,
    )
    monkeypatch.setenv("RANK", "2")  # type: ignore[attr-defined]
    assert is_rank_zero() is False
    # Sanity: clearing env vars makes the dist-reported rank visible.
    monkeypatch.delenv("RANK", raising=False)  # type: ignore[attr-defined]
    monkeypatch.delenv("LOCAL_RANK", raising=False)  # type: ignore[attr-defined]
    # With env cleared, dist mock returns 0 → True.
    assert is_rank_zero() is True
