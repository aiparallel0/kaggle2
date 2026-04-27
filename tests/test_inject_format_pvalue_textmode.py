"""test_inject_format_pvalue_textmode.py — _format_pvalue/_format_lr are text-mode-safe.

Regression guard: these functions previously returned raw math content
(``3.2\\times 10^{-5}``) without an outer ``\\ensuremath{}``.  When the
same key appears in a text-mode context (e.g. ``McNemar $p=$\\VAR{mcnemar_p}``
where the ``$`` closes *before* the VAR), tectonic aborts with
``Missing $ inserted``.  The fix wraps output in ``\\ensuremath{...}`` so
the content is safe in both math and text contexts.
"""
from __future__ import annotations

from report.inject import _format_lr, _format_pvalue


def test_format_pvalue_small_uses_ensuremath() -> None:
    """Small p-values must be wrapped in ``\\ensuremath{}`` (not bare math)."""
    out = _format_pvalue(3e-5)
    assert out.startswith("\\ensuremath{"), out
    assert "\\times" in out
    assert "10^{-5}" in out


def test_format_pvalue_zero_uses_ensuremath() -> None:
    """Underflow p=0.0 must be wrapped in ``\\ensuremath{}``."""
    out = _format_pvalue(0.0)
    assert out.startswith("\\ensuremath{"), out
    assert "10^{-12}" in out


def test_format_pvalue_large_is_plain_decimal() -> None:
    """Values >= 1e-4 render as plain decimal (no math markup needed)."""
    out = _format_pvalue(0.0432)
    assert out == "0.0432"
    assert "\\times" not in out
    assert "\\ensuremath" not in out


def test_format_pvalue_no_bare_times_outside_ensuremath() -> None:
    """No bare ``\\times`` must appear outside ``\\ensuremath``."""
    for val in (3e-5, 1.5e-10, 0.0):
        out = _format_pvalue(val)
        # Strip \ensuremath{...} blocks, then check no \times remains.
        import re
        stripped = re.sub(r"\\ensuremath\{[^}]*(?:\{[^}]*\}[^}]*)?\}", "", out)
        assert "\\times" not in stripped, f"bare \\times in {out!r}"


def test_format_lr_small_uses_ensuremath() -> None:
    """Learning rates must be wrapped in ``\\ensuremath{}``."""
    out = _format_lr(5e-5)
    assert out.startswith("\\ensuremath{"), out
    assert "\\times" in out


def test_format_lr_zero_is_plain() -> None:
    """lr=0.0 is the trivial case — no math markup."""
    assert _format_lr(0.0) == "0"
