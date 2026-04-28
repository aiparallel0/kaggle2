"""PR-A / T-H1 — anti-regression invariants for the eval stage."""
from __future__ import annotations

import logging

from core.types import Metrics


def _make_metrics(global_f1: float) -> Metrics:
    fields = ("company", "address", "date", "total")
    return Metrics(
        global_f1=global_f1,
        global_ned=global_f1,
        global_em=global_f1,
        per_field_f1={f: global_f1 for f in fields},
        per_field_ned={f: global_f1 for f in fields},
        per_field_em={f: global_f1 for f in fields},
    )


def test_pipeline_below_gtocr_warns(caplog: object) -> None:
    """Hybrid F1 < gtocr_rb F1 by > epsilon must surface a WARNING."""
    from stages.common import assert_hybrid_beats_gtocr_rulebased

    pipeline = _make_metrics(global_f1=0.55)
    gtocr = _make_metrics(global_f1=0.73)
    cap = caplog  # type: ignore[assignment]
    with cap.at_level(logging.WARNING, logger="kaggle2"):  # type: ignore[attr-defined]
        assert_hybrid_beats_gtocr_rulebased(pipeline, gtocr)
    text = "\n".join(r.message for r in cap.records)  # type: ignore[attr-defined]
    assert "F1=" in text
    assert "gtocr_rulebased_f1" in text


def test_pipeline_above_gtocr_silent(caplog: object) -> None:
    from stages.common import assert_hybrid_beats_gtocr_rulebased

    pipeline = _make_metrics(global_f1=0.85)
    gtocr = _make_metrics(global_f1=0.73)
    cap = caplog  # type: ignore[assignment]
    with cap.at_level(logging.WARNING, logger="kaggle2"):  # type: ignore[attr-defined]
        assert_hybrid_beats_gtocr_rulebased(pipeline, gtocr)
    warns = [r for r in cap.records  # type: ignore[attr-defined]
             if r.levelno >= logging.WARNING]
    assert not warns
