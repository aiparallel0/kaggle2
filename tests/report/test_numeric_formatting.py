"""Tests for Item 12: Numeric formatting consistency."""
from report.inject_format import apply_directive, apply_formatters


def test_pct1_directive() -> None:
    """pct1 formats as percentage with 1 decimal."""
    assert apply_directive(0.842, "pct1") == "84.2\\%"
    assert apply_directive(0.5, "pct1") == "50.0\\%"


def test_pct0_directive() -> None:
    """pct0 formats as integer percentage."""
    assert apply_directive(0.847, "pct0") == "85\\%"


def test_ms_directive() -> None:
    """ms formats as milliseconds integer."""
    assert apply_directive(142.3, "ms") == "142\\,ms"


def test_usd_directive() -> None:
    """usd formats as dollar with 2 decimals."""
    assert apply_directive(1.234, "usd") == "\\$1.23"


def test_sig3_directive() -> None:
    """sig3 formats with 3 significant figures."""
    result = apply_directive(0.1234, "sig3")
    assert result is not None
    assert "0.123" in result


def test_p_value_directive_small() -> None:
    """p directive formats small values in scientific notation."""
    result = apply_directive(3.2e-5, "p")
    assert result is not None
    assert "\\times" in result or "10^" in result


def test_p_value_directive_normal() -> None:
    """p directive formats normal values with 4 decimals."""
    assert apply_directive(0.0432, "p") == "0.0432"


def test_mean_std_directive() -> None:
    """mean_std_pct1 combines mean and std with ± separator."""
    metrics = {"f1_mean": 0.85, "f1_std": 0.02}
    template = r"\VAR{f1:mean_std_pct1}"
    result = apply_formatters(template, metrics)
    assert "85.0" in result
    assert "2.0" in result
    assert "\\ensuremath{\\pm}" in result


def test_mean_std_missing_std() -> None:
    """mean_std with missing std shows only mean."""
    metrics = {"f1_mean": 0.85}
    template = r"\VAR{f1:mean_std_pct1}"
    result = apply_formatters(template, metrics)
    assert "85.0" in result
    assert "\\pm" not in result
