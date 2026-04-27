"""Tests for Item 4: CI bounds assertion (lo ≤ point ≤ hi)."""
import pytest

from report.missing import assert_ci_bounds_valid


def test_valid_ci_bounds_pass() -> None:
    """Valid bounds should not raise."""
    metrics = {
        "donut_f1_company": 0.85,
        "donut_f1_company_ci_lo": 0.80,
        "donut_f1_company_ci_hi": 0.90,
    }
    assert_ci_bounds_valid(metrics)  # no raise


def test_missing_keys_skipped() -> None:
    """Missing keys are silently skipped."""
    metrics = {"donut_f1_company": 0.85}  # no CI bounds
    assert_ci_bounds_valid(metrics)  # no raise


def test_lo_greater_than_point_raises() -> None:
    """ci_lo > point should raise."""
    metrics = {
        "donut_f1_company": 0.80,
        "donut_f1_company_ci_lo": 0.85,  # > point
        "donut_f1_company_ci_hi": 0.90,
    }
    with pytest.raises(ValueError, match="ci_lo=0.85.* > point"):
        assert_ci_bounds_valid(metrics)


def test_hi_less_than_point_raises() -> None:
    """ci_hi < point should raise."""
    metrics = {
        "pipeline_f1_date": 0.90,
        "pipeline_f1_date_ci_lo": 0.80,
        "pipeline_f1_date_ci_hi": 0.85,  # < point
    }
    with pytest.raises(ValueError, match="ci_hi=0.85.* < point"):
        assert_ci_bounds_valid(metrics)


def test_tolerance_within_1e6() -> None:
    """Values within tolerance (1e-6) should pass."""
    metrics = {
        "donut_f1_total": 0.8500001,
        "donut_f1_total_ci_lo": 0.85,
        "donut_f1_total_ci_hi": 0.85,
    }
    assert_ci_bounds_valid(metrics)  # tolerance allows tiny deviation


def test_all_fields_and_systems_checked() -> None:
    """All 8 field/system combinations are checked."""
    metrics = {}
    for sys in ("donut", "pipeline"):
        for field in ("company", "date", "address", "total"):
            metrics[f"{sys}_f1_{field}"] = 0.80
            metrics[f"{sys}_f1_{field}_ci_lo"] = 0.70
            metrics[f"{sys}_f1_{field}_ci_hi"] = 0.90
    assert_ci_bounds_valid(metrics)  # all valid


def test_non_numeric_skipped() -> None:
    """Non-numeric values are skipped."""
    metrics = {
        "donut_f1_company": "N/A",
        "donut_f1_company_ci_lo": 0.80,
        "donut_f1_company_ci_hi": 0.90,
    }
    assert_ci_bounds_valid(metrics)  # skipped due to non-numeric point
