"""PR-A / T-H2 — McNemar exact-binomial guard for the c=0 pathology.

The asymptotic chi-squared form of McNemar collapses to machine epsilon
when the discordant counts (b, c) are very unequal — e.g. when one
system almost always wins where the other loses.  The repo's
``core.statistics.mcnemar`` uses the exact binomial form
(``binomtest(min(b,c), n=b+c, p=0.5)``) which stays finite and
hand-computable.

This test pins that behaviour so the asymptotic regression cannot
return.
"""
from __future__ import annotations

import math

from core.statistics import mcnemar


def test_mcnemar_finite_when_c_equals_zero() -> None:
    """All discordances on one side: exact p must be finite + non-zero.

    Construct a 30-receipt synthetic pair where pipeline wins on 8
    receipts and donut wins on 0 — the asymptotic χ² form returns
    a value of ``8`` and a p-value at the lower edge of float64;
    the exact binomial returns ``2 * binom.cdf(0; 8, 0.5) = 2/2^7
    = 1/128 ≈ 0.0078125``.
    """
    a = [True] * 22 + [False] * 8        # donut: 22 ✓, 8 ✗
    b = [True] * 22 + [True] * 8         # pipeline: 30 ✓
    p = mcnemar(a, b)
    assert math.isfinite(p)
    # The exact binomial 2-sided p with b=8, c=0 is 2 * 0.5^8 = 1/128.
    expected = 2.0 * (0.5 ** 8)
    assert math.isclose(p, expected, rel_tol=1e-6)


def test_mcnemar_balanced_returns_one() -> None:
    """Equal discordant counts: exact 2-sided p must equal 1.0."""
    a = [True] * 4 + [False] * 4
    b = [False] * 4 + [True] * 4
    p = mcnemar(a, b)
    assert math.isclose(p, 1.0, rel_tol=1e-6)
