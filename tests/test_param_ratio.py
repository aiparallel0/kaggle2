"""Tests for Item 5 / Audit C2: param_ratio_phrase banded resolver."""
from unittest.mock import MagicMock


def test_param_ratio_phrase_one_third() -> None:
    """Ratio in (0.30, 0.40] should yield ``one-third``."""
    from report.combine import merge_pipeline_diagnostics

    config = MagicMock()
    config.output_dir = "/nonexistent"  # No files to load
    metrics: dict[str, object] = {
        "donut_params_m": 200,
        "pipeline_params_m": 65,  # 32.5%
    }
    merge_pipeline_diagnostics(config, metrics)
    assert metrics.get("param_ratio_phrase") == "one-third"
    # numeric is now a LaTeX-safe percent string ("32.5\\%")
    assert metrics.get("param_ratio_numeric") == "32.5\\%"


def test_param_ratio_phrase_one_quarter() -> None:
    """Ratio in (0.20, 0.30] should yield ``one-quarter`` (Audit C2)."""
    from report.combine import merge_pipeline_diagnostics

    config = MagicMock()
    config.output_dir = "/nonexistent"
    metrics: dict[str, object] = {
        "donut_params_m": 200,
        "pipeline_params_m": 50,  # 25.0%
    }
    merge_pipeline_diagnostics(config, metrics)
    assert metrics.get("param_ratio_phrase") == "one-quarter"
    assert metrics.get("param_ratio_numeric") == "25.0\\%"


def test_param_ratio_phrase_one_fifth() -> None:
    """Ratio ≤ 0.20 should yield ``one-fifth``."""
    from report.combine import merge_pipeline_diagnostics

    config = MagicMock()
    config.output_dir = "/nonexistent"
    metrics: dict[str, object] = {
        "donut_params_m": 100,
        "pipeline_params_m": 20,  # 20.0%
    }
    merge_pipeline_diagnostics(config, metrics)
    assert metrics.get("param_ratio_phrase") == "one-fifth"
    assert metrics.get("param_ratio_numeric") == "20.0\\%"


def test_param_ratio_phrase_high_ratio() -> None:
    """Ratio in (0.40, 0.55] should yield ``roughly half``."""
    from report.combine import merge_pipeline_diagnostics

    config = MagicMock()
    config.output_dir = "/nonexistent"
    metrics: dict[str, object] = {
        "donut_params_m": 100,
        "pipeline_params_m": 50,  # 50.0%
    }
    merge_pipeline_diagnostics(config, metrics)
    assert metrics.get("param_ratio_phrase") == "roughly half"


def test_missing_ok_keys_include_param_ratio() -> None:
    """param_ratio keys should be in MISSING_OK_KEYS."""
    from report.missing import MISSING_OK_KEYS
    assert "param_ratio_phrase" in MISSING_OK_KEYS
    assert "param_ratio_numeric" in MISSING_OK_KEYS
