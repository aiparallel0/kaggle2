"""Tests for the ``:pct1`` / ``:ms`` / ``:usd`` directive DSL."""
from __future__ import annotations

from report.inject_format import apply_directive, apply_formatters


def test_pct_directives() -> None:
    assert apply_directive(0.842, "pct1") == "84.2\\%"
    assert apply_directive(0.842, "pct2") == "84.20\\%"
    assert apply_directive(0.842, "pct0") == "84\\%"


def test_ms_directives() -> None:
    assert apply_directive(142.3, "ms") == "142\\,ms"
    assert apply_directive(142.3, "ms1") == "142.3\\,ms"


def test_usd_directives() -> None:
    assert apply_directive(1.234, "usd") == "\\$1.23"
    assert apply_directive(0.001234, "usd4") == "\\$0.0012"


def test_wh_and_gb_directives() -> None:
    assert apply_directive(123.45, "wh") == "123\\,Wh"
    assert apply_directive(12.34, "gb1") == "12.3\\,GiB"


def test_sig_fig_directives() -> None:
    assert apply_directive(123.4567, "sig3") == "123"
    assert apply_directive(123.4567, "sig4") == "123.5"


def test_int_directive() -> None:
    assert apply_directive(1234567.0, "int") == "1,234,567"


def test_unknown_directive_returns_none() -> None:
    assert apply_directive(1.0, "bananas") is None


def test_non_numeric_value_returns_none() -> None:
    assert apply_directive([1, 2, 3], "pct1") is None
    assert apply_directive("not a float", "pct1") is None


def test_string_numeric_value_is_parsed() -> None:
    assert apply_directive("0.5", "pct1") == "50.0\\%"


def test_apply_formatters_end_to_end() -> None:
    template = "F1=\\VAR{f1:pct1}, lat=\\VAR{ms:ms}, usd=\\VAR{cost:usd}"
    metrics: dict[str, object] = {"f1": 0.842, "ms": 142.3, "cost": 1.234}
    out = apply_formatters(template, metrics)
    assert "84.2\\%" in out
    assert "142\\,ms" in out
    assert "\\$1.23" in out


def test_apply_formatters_preserves_unknown_keys() -> None:
    """Missing key → placeholder intact so inject_results' audit sees it."""
    template = "\\VAR{unknown_key:pct1}"
    out = apply_formatters(template, {})
    assert out == "\\VAR{unknown_key:pct1}"


def test_apply_formatters_unknown_directive_falls_back() -> None:
    """Unknown directive → strip directive, let base injector format."""
    template = "\\VAR{key:banana}"
    out = apply_formatters(template, {"key": 0.5})
    assert out == "\\VAR{key}"
