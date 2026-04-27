"""Tests for Item 4: CI bounds assertion (lo ≤ mean ≤ hi)."""
import pytest

from report.missing import assert_ci_bounds_valid


def test_valid_ci_bounds_pass() -> None:
    """Valid bounds should not raise."""
    metrics = {
        "donut_f1_company_mean": 0.85,
        "donut_f1_company_ci_lo": 0.80,
        "donut_f1_company_ci_hi": 0.90,
    }
    assert_ci_bounds_valid(metrics)  # no raise


def test_missing_keys_skipped() -> None:
    """Missing keys are silently skipped."""
    metrics = {"donut_f1_company": 0.85}  # no _mean or CI bounds
    assert_ci_bounds_valid(metrics)  # no raise


def test_lo_greater_than_mean_raises() -> None:
    """ci_lo > mean should raise."""
    metrics = {
        "donut_f1_company_mean": 0.80,
        "donut_f1_company_ci_lo": 0.85,  # > mean
        "donut_f1_company_ci_hi": 0.90,
    }
    with pytest.raises(ValueError, match="ci_lo=0.85.* > mean"):
        assert_ci_bounds_valid(metrics)


def test_hi_less_than_mean_raises() -> None:
    """ci_hi < mean should raise."""
    metrics = {
        "pipeline_f1_date_mean": 0.90,
        "pipeline_f1_date_ci_lo": 0.80,
        "pipeline_f1_date_ci_hi": 0.85,  # < mean
    }
    with pytest.raises(ValueError, match="ci_hi=0.85.* < mean"):
        assert_ci_bounds_valid(metrics)


def test_tolerance_within_1e6() -> None:
    """Values within tolerance (1e-6) should pass."""
    metrics = {
        "donut_f1_total_mean": 0.8500001,
        "donut_f1_total_ci_lo": 0.85,
        "donut_f1_total_ci_hi": 0.85,
    }
    assert_ci_bounds_valid(metrics)  # tolerance allows tiny deviation


def test_all_fields_and_systems_checked() -> None:
    """All 8 field/system combinations are checked."""
    metrics = {}
    for sys in ("donut", "pipeline"):
        for field in ("company", "date", "address", "total"):
            metrics[f"{sys}_f1_{field}_mean"] = 0.80
            metrics[f"{sys}_f1_{field}_ci_lo"] = 0.70
            metrics[f"{sys}_f1_{field}_ci_hi"] = 0.90
    assert_ci_bounds_valid(metrics)  # all valid


def test_non_numeric_skipped() -> None:
    """Non-numeric values are skipped."""
    metrics = {
        "donut_f1_company_mean": "N/A",
        "donut_f1_company_ci_lo": 0.80,
        "donut_f1_company_ci_hi": 0.90,
    }
    assert_ci_bounds_valid(metrics)  # skipped due to non-numeric mean


def test_point_outside_ci_but_mean_inside_does_not_raise() -> None:
    """Regression: multi-seed run where last-seed point is outside CI but mean is inside.

    This mirrors the exact failure from the bug report where donut_f1_company
    point=0.8818 fell outside ci_hi=0.8133 while the mean was well within bounds.
    assert_ci_bounds_valid must not raise in this scenario.
    """
    metrics = {
        # bare point value from last seed — outside the CI interval
        "donut_f1_company": 0.8818,
        # mean across seeds — inside the CI interval
        "donut_f1_company_mean": 0.78,
        "donut_f1_company_ci_lo": 0.70,
        "donut_f1_company_ci_hi": 0.8133,
    }
    assert_ci_bounds_valid(metrics)  # must not raise


def test_canonical_reference_run_brackets_point_estimate() -> None:
    """Audit B1 regression: on canonical n=1, the bare point IS bracketed.

    Single-seed runs write ``<sys>_f1_<field>`` and (when extended-metrics
    ran) ``_ci_lo`` / ``_ci_hi`` from the same per-image bundle.  Because
    the producer (B1 fix) now bootstraps the *global statistic* — the
    same arithmetic-mean estimator the bare point uses — the bare point
    must satisfy ``ci_lo ≤ point ≤ ci_hi``.  When ``_mean`` is missing
    or equals the point (n=1), the assertion checks the bare point.
    """
    metrics = {
        "donut_f1_date": 0.9,
        "donut_f1_date_ci_lo": 0.85,
        "donut_f1_date_ci_hi": 0.95,
    }
    assert_ci_bounds_valid(metrics)  # must not raise — point inside CI


def test_canonical_n1_point_outside_ci_raises() -> None:
    """Audit B1: n=1 with point outside CI is an estimator mismatch — raise."""
    import pytest as _pytest
    metrics = {
        # n=1: mean equals point, both 0.92 — outside [0.70, 0.85]
        "donut_f1_date": 0.92,
        "donut_f1_date_mean": 0.92,
        "donut_f1_date_ci_lo": 0.70,
        "donut_f1_date_ci_hi": 0.85,
    }
    with _pytest.raises(ValueError):
        assert_ci_bounds_valid(metrics)
