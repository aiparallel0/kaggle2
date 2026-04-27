"""Tests for Item 5: param_ratio_phrase single-source computation."""
from unittest.mock import MagicMock


def test_param_ratio_phrase_one_third() -> None:
    """Ratio around 1/3 (0.29-0.37) should yield 'roughly one-third'."""
    from report.combine import merge_pipeline_diagnostics

    config = MagicMock()
    config.output_dir = "/nonexistent"  # No files to load
    metrics: dict[str, object] = {
        "donut_params_m": 200,
        "pipeline_params_m": 65,
    }
    merge_pipeline_diagnostics(config, metrics)
    assert metrics.get("param_ratio_phrase") == "roughly one-third"
    ratio = metrics.get("param_ratio_numeric")
    assert ratio is not None
    assert 0.29 <= float(ratio) <= 0.37


def test_param_ratio_phrase_low_ratio() -> None:
    """Ratio < 0.29 should yield percent format."""
    from report.combine import merge_pipeline_diagnostics

    config = MagicMock()
    config.output_dir = "/nonexistent"
    metrics: dict[str, object] = {
        "donut_params_m": 100,
        "pipeline_params_m": 20,  # 20% ratio
    }
    merge_pipeline_diagnostics(config, metrics)
    assert metrics.get("param_ratio_phrase") == "$\\approx$20%"


def test_param_ratio_phrase_high_ratio() -> None:
    """Ratio > 0.37 should yield percent format."""
    from report.combine import merge_pipeline_diagnostics

    config = MagicMock()
    config.output_dir = "/nonexistent"
    metrics: dict[str, object] = {
        "donut_params_m": 100,
        "pipeline_params_m": 50,  # 50% ratio
    }
    merge_pipeline_diagnostics(config, metrics)
    assert metrics.get("param_ratio_phrase") == "$\\approx$50%"


def test_missing_ok_keys_include_param_ratio() -> None:
    """param_ratio keys should be in MISSING_OK_KEYS."""
    from report.missing import MISSING_OK_KEYS
    assert "param_ratio_phrase" in MISSING_OK_KEYS
    assert "param_ratio_numeric" in MISSING_OK_KEYS
