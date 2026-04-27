"""PR-A / T-G3 — determinism guard for the assigner train loop.

Any new ``DataLoader`` must use the seeded ``worker_init_fn`` +
``torch.Generator`` pattern.  This test pins the high-level invariant:
calling :func:`core.seed.seed_everything` twice with the same seed
must leave torch / numpy / random in identical states so two
back-to-back training runs are bit-for-bit reproducible.

Skipped on no-CUDA boxes for the GPU-side guard; the CPU+torch
guard always runs.
"""
from __future__ import annotations

import random

import pytest


def test_seed_everything_pins_python_random() -> None:
    from core.seed import seed_everything

    seed_everything(42)
    a = [random.random() for _ in range(8)]
    seed_everything(42)
    b = [random.random() for _ in range(8)]
    assert a == b


def test_seed_everything_pins_numpy() -> None:
    np = pytest.importorskip("numpy")
    from core.seed import seed_everything

    seed_everything(7)
    a = np.random.rand(8).tolist()
    seed_everything(7)
    b = np.random.rand(8).tolist()
    assert a == b


def test_seed_everything_pins_torch_cpu() -> None:
    torch = pytest.importorskip("torch")
    from core.seed import seed_everything

    seed_everything(13)
    a = torch.randn(8).tolist()
    seed_everything(13)
    b = torch.randn(8).tolist()
    assert a == b


def test_seed_everything_pins_torch_cuda() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available on this box.")
    from core.seed import seed_everything

    seed_everything(101)
    a = torch.randn(8, device="cuda").cpu().tolist()
    seed_everything(101)
    b = torch.randn(8, device="cuda").cpu().tolist()
    assert a == b
