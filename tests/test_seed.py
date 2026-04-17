"""Unit tests for core.seed — deterministic seeding across stdlib + torch."""
from __future__ import annotations

import random

from core.seed import seed_everything


def test_python_random_is_deterministic() -> None:
    seed_everything(42)
    a = [random.random() for _ in range(5)]
    seed_everything(42)
    b = [random.random() for _ in range(5)]
    assert a == b


def test_different_seeds_produce_different_sequences() -> None:
    seed_everything(1)
    a = [random.random() for _ in range(5)]
    seed_everything(2)
    b = [random.random() for _ in range(5)]
    assert a != b
