"""test_mcnemar.py — known 2×2 table matches a hand-calculated reference."""
from __future__ import annotations


def test_mcnemar_symmetric() -> None:
    """b == c → p-value = 1.0 (no evidence of difference)."""
    from core.statistics import mcnemar

    a = [True, True, False, False] * 10
    b = [True, False, True, False] * 10
    p = mcnemar(a, b)
    # b == c == 10, so two-tailed p ≈ 1.0
    assert abs(p - 1.0) < 1e-6, p


def test_mcnemar_perfect_disagreement() -> None:
    """All errors on one side → very small p."""
    from core.statistics import mcnemar

    # A always right where B is wrong, B never right where A is wrong
    n = 30
    a = [True] * n + [False] * 0
    b = [False] * n + [True] * 0
    p = mcnemar(a, b)
    assert p < 0.001, f"Expected p < 0.001 for extreme disagreement, got {p}"


def test_mcnemar_both_same() -> None:
    """Both systems agree on all images → p = 1.0 (b + c = 0)."""
    from core.statistics import mcnemar

    same = [True] * 50 + [False] * 13
    p = mcnemar(same, same)
    assert p == 1.0, p


def test_mcnemar_known_table() -> None:
    """Small known table: b=6, c=2.  Exact p via Binom(8, 0.5) two-tail ≈ 0.289."""
    import math

    from core.statistics import mcnemar

    # Construct boolean vectors that yield b=6, c=2
    # b = A correct, B wrong  (6 images)
    # c = A wrong,   B correct (2 images)
    a = [True] * 6 + [False] * 2 + [True] * 40 + [False] * 5
    b = [False] * 6 + [True] * 2 + [True] * 40 + [False] * 5
    p = mcnemar(a, b)
    # Manual: n=8, k_obs=min(6,2)=2
    # P(X<=2|Binom(8,0.5)) = sum C(8,k)*(0.5)^8 for k=0..2
    expected_one_tail = sum(
        math.comb(8, k) * (0.5**8) for k in range(3)
    )
    expected = min(1.0, 2.0 * expected_one_tail)
    assert abs(p - expected) < 1e-9, f"p={p} expected={expected}"
