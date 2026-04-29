r"""test_inject_format_value_latex_escape.py — _format_value escapes LaTeX special chars.

Regression guard: ``_format_value`` previously fell through to bare ``str(value)``
for non-numeric metrics, emitting unescaped ``_`` (and ``&``, ``%``, etc.) into
text mode.  This caused tectonic to abort with ``Missing $ inserted`` at line 1468
of ``paper_filled.tex`` in run ``20260427T071206Z-fd9d7b0`` where
``\VAR{test_set_kind}`` resolved to the literal ``canonical_347``.

The fix applies ``_latex_escape_text`` to all string fallthrough returns.
Numeric formatters and explicit LaTeX literals (``\ensuremath{...}``,
``\MissingCell{...}``) are already safe and must NOT be re-escaped.
"""
from __future__ import annotations

from report.inject import _format_value, inject_results


def test_underscore_in_string_value_is_escaped() -> None:
    """``test_set_kind = "canonical_347"`` must not contain a bare ``_``."""
    import re
    out = _format_value("test_set_kind", "canonical_347", {})
    assert r"\_" in out, f"expected escaped underscore in: {out!r}"
    # No bare underscore (not preceded by backslash).
    assert not re.search(r"(?<!\\)_", out), f"bare _ in: {out!r}"


def test_ampersand_in_string_value_is_escaped() -> None:
    out = _format_value("dataset_name", "foo&bar", {})
    assert "&" not in out.replace(r"\&", ""), f"bare & in: {out!r}"
    assert r"\&" in out


def test_percent_in_string_value_is_escaped() -> None:
    out = _format_value("split", "100%", {})
    assert "%" not in out.replace(r"\%", ""), f"bare % in: {out!r}"
    assert r"\%" in out


def test_numeric_float_value_is_unchanged() -> None:
    """Floats must still render as 4-decimal strings with no escaping."""
    out = _format_value("donut_f1", 0.85, {})
    assert out == "0.8500"
    assert r"\_" not in out


def test_format_pvalue_ensuremath_not_re_escaped() -> None:
    """``mcnemar_p`` branch returns ``\\ensuremath{...}`` — must not be escaped."""
    out = _format_value("mcnemar_p", 3e-5, {})
    assert out.startswith("\\ensuremath{"), out
    assert r"\textbackslash" not in out, f"backslash was escaped: {out!r}"


def test_format_lr_ensuremath_not_re_escaped() -> None:
    """LR branch returns ``\\ensuremath{...}`` — must not be escaped."""
    out = _format_value("lr", 5e-5, {})
    assert out.startswith("\\ensuremath{"), out
    assert r"\textbackslash" not in out, f"backslash was escaped: {out!r}"


def test_mean_std_ensuremath_not_re_escaped() -> None:
    """mean_std branch returns ``\\ensuremath{\\pm}`` — must not be escaped."""
    metrics = {
        "donut_f1": 0.85,
        "donut_f1_mean": 0.85,
        "donut_f1_std": 0.02,
        "seeds_used": [42, 43],
    }
    out = _format_value("donut_f1", 0.85, metrics)
    assert "\\ensuremath" in out, out
    assert r"\textbackslash" not in out, f"backslash was escaped: {out!r}"


def test_seeds_used_not_double_escaped() -> None:
    """``seeds_used`` branch returns early; seed IDs (ints) are safe, no change."""
    out = _format_value("seeds_used", [42, 43, 44], {})
    assert out == "3 seeds (seeds 42, 43, 44)"


def test_inject_results_integration_canonical_347() -> None:
    """Integration: ``\\VAR{test_set_kind}`` → ``canonical\\_347`` in output."""
    template = r"This run used \VAR{test_set_kind} (\VAR{test_set_size} test images)."
    metrics: dict[str, object] = {"test_set_kind": "canonical_347", "test_set_size": 347}
    out = inject_results(template, metrics)
    assert "canonical_347" not in out, f"bare canonical_347 in: {out!r}"
    assert r"canonical\_347" in out, f"expected canonical\\_347 in: {out!r}"
