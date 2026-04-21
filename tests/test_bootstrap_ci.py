"""test_bootstrap_ci.py — CI width shrinks with larger n."""
from __future__ import annotations

import random


def test_ci_width_shrinks() -> None:
    random.seed(42)
    from core.statistics import bootstrap_ci

    small = [random.random() > 0.4 for _ in range(20)]
    large = [random.random() > 0.4 for _ in range(200)]

    lo_s, hi_s = bootstrap_ci(small, n_iter=2000)
    lo_l, hi_l = bootstrap_ci(large, n_iter=2000)

    width_small = hi_s - lo_s
    width_large = hi_l - lo_l
    assert width_small > width_large, (
        f"CI width should shrink: small={width_small:.4f} large={width_large:.4f}"
    )


def test_ci_empty() -> None:
    from core.statistics import bootstrap_ci

    lo, hi = bootstrap_ci([])
    assert lo == 0.0 and hi == 0.0


def test_ci_all_correct() -> None:
    random.seed(0)
    from core.statistics import bootstrap_ci

    lo, hi = bootstrap_ci([True] * 100, n_iter=500)
    assert lo == 1.0 and hi == 1.0


def test_ci_all_wrong() -> None:
    random.seed(0)
    from core.statistics import bootstrap_ci

    lo, hi = bootstrap_ci([False] * 100, n_iter=500)
    assert lo == 0.0 and hi == 0.0
