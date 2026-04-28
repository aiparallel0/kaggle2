"""Tests for KAGGLE2_F1_WARN_ONLY env-var warn-only path in validate_f1."""
from __future__ import annotations

import os

import pytest

from core.errors import EvalError
from core.types import Metrics
from core.validate import validate_f1

_LOW_F1 = Metrics(
    global_f1=0.009,
    global_ned=0.0,
    global_em=0.0,
    per_field_f1={"company": 0.0, "date": 0.0, "address": 0.0, "total": 0.0},
    per_field_ned={"company": 0.0, "date": 0.0, "address": 0.0, "total": 0.0},
    per_field_em={"company": 0.0, "date": 0.0, "address": 0.0, "total": 0.0},
)

_GOOD_F1 = Metrics(
    global_f1=0.85,
    global_ned=0.9,
    global_em=0.7,
    per_field_f1={"company": 0.85, "date": 0.85, "address": 0.80, "total": 0.90},
    per_field_ned={"company": 0.9, "date": 0.9, "address": 0.85, "total": 0.92},
    per_field_em={"company": 0.7, "date": 0.7, "address": 0.65, "total": 0.75},
)


def test_donut_low_f1_raises_by_default() -> None:
    """Without the env var, low DONUT F1 must raise EvalError."""
    os.environ.pop("KAGGLE2_F1_WARN_ONLY", None)
    with pytest.raises(EvalError, match="DONUT F1=0.0090 < 0.50"):
        validate_f1(_LOW_F1, "donut")


def test_donut_low_f1_warn_only_does_not_raise() -> None:
    """With KAGGLE2_F1_WARN_ONLY=1, low DONUT F1 must not raise."""
    os.environ["KAGGLE2_F1_WARN_ONLY"] = "1"
    try:
        validate_f1(_LOW_F1, "donut")  # must return normally
    finally:
        os.environ.pop("KAGGLE2_F1_WARN_ONLY", None)


def test_donut_good_f1_never_raises() -> None:
    """Healthy DONUT F1 must not raise regardless of the env var."""
    os.environ.pop("KAGGLE2_F1_WARN_ONLY", None)
    validate_f1(_GOOD_F1, "donut")  # must not raise


def test_warn_only_zero_disables_override() -> None:
    """KAGGLE2_F1_WARN_ONLY=0 (explicit default) still raises."""
    os.environ["KAGGLE2_F1_WARN_ONLY"] = "0"
    try:
        with pytest.raises(EvalError):
            validate_f1(_LOW_F1, "donut")
    finally:
        os.environ.pop("KAGGLE2_F1_WARN_ONLY", None)


def test_pipeline_zero_f1_unaffected_by_warn_only() -> None:
    """KAGGLE2_F1_WARN_ONLY only applies to 'donut', not 'pipeline'."""
    os.environ["KAGGLE2_F1_WARN_ONLY"] = "1"
    zero_pipeline = Metrics(
        global_f1=0.0,
        global_ned=0.0,
        global_em=0.0,
        per_field_f1={"company": 0.0, "date": 0.0, "address": 0.0, "total": 0.0},
        per_field_ned={"company": 0.0, "date": 0.0, "address": 0.0, "total": 0.0},
        per_field_em={"company": 0.0, "date": 0.0, "address": 0.0, "total": 0.0},
    )
    try:
        with pytest.raises(EvalError, match="Pipeline F1=0.0"):
            validate_f1(zero_pipeline, "pipeline")
    finally:
        os.environ.pop("KAGGLE2_F1_WARN_ONLY", None)
