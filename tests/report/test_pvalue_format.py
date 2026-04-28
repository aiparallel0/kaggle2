"""test_pvalue_format.py — McNemar p-value renders without the ``%.4f`` trap.

Review item S5: the headline statistical claim shipped as ``p=0.0000``
adjacent to a bootstrap CI that "straddles zero".  Either the bootstrap
or the McNemar print precision was wrong; ``round(p, 4)`` for a true
``p ≈ 3e-5`` rounds to ``0.0000`` and is the more likely culprit.

This test pins the formatter contract: full-precision floats in,
scientific-notation LaTeX out for tiny p-values; ``%.4f`` for moderate
ones.
"""
from __future__ import annotations

from report.inject import inject_results


def test_tiny_pvalue_renders_scientific() -> None:
    out = inject_results("\\VAR{mcnemar_p}", {"mcnemar_p": 3.2e-5})
    assert "10^{-5}" in out, out
    assert "0.0000" not in out


def test_moderate_pvalue_renders_decimal() -> None:
    out = inject_results("\\VAR{mcnemar_p}", {"mcnemar_p": 0.0123})
    assert "0.0123" in out


def test_zero_pvalue_renders_lower_bound() -> None:
    """A literal ``0.0`` (numerical underflow) renders with a lower bound."""
    out = inject_results("\\VAR{mcnemar_p}", {"mcnemar_p": 0.0})
    assert "10^{-12}" in out, out
