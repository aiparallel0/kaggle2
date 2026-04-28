"""Tests for :mod:`stages.eval_producers` — real-data-only writers."""
from __future__ import annotations

import json
from pathlib import Path

from core.types import Field, Metrics, Prediction, Receipt
from stages.eval_producers import (
    emit_all,
    write_errors_jsonl,
    write_extended_metrics,
    write_preds_jsonl,
)


def _mkreceipt(tmp: Path, stem: str, **fields: str) -> Receipt:
    """Build a minimal Receipt with stub image_path."""
    img = tmp / f"{stem}.jpg"
    img.write_bytes(b"")
    return Receipt(
        image_path=img,
        fields=[Field(name=k, value=v) for k, v in fields.items()],
    )


def _mkpred(stem: str, **fields: str) -> Prediction:
    return Prediction(
        receipt_id=stem,
        fields=[Field(name=k, value=v) for k, v in fields.items()],
    )


def test_write_preds_jsonl_one_line_per_pair(tmp_path: Path) -> None:
    rec = _mkreceipt(tmp_path, "r1", company="ACME", total="10.00")
    pred = _mkpred("r1", company="ACME", total="10.00")
    path = tmp_path / "preds.jsonl"
    n = write_preds_jsonl(path, [pred], [rec], ("company", "total"), "donut")
    assert n == 1
    row = json.loads(path.read_text().splitlines()[0])
    assert row["image_id"] == "r1"
    assert row["model"] == "donut"
    assert row["per_field_exact"]["company"] is True
    assert row["per_field_exact"]["total"] is True


def test_write_errors_jsonl_emits_one_row_per_field(tmp_path: Path) -> None:
    rec = _mkreceipt(tmp_path, "r1", company="ACME", total="10.00")
    pred = _mkpred("r1", company="", total="10.00")
    path = tmp_path / "errors.jsonl"
    n = write_errors_jsonl(path, [pred], [rec], ("company", "total"), "donut")
    assert n == 2
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert {r["field"] for r in rows} == {"company", "total"}
    # Missing company → should be a miss category other than "correct".
    company_row = next(r for r in rows if r["field"] == "company")
    assert company_row["category"] != "correct"


def test_write_extended_metrics_prefixes_keys(tmp_path: Path) -> None:
    from core.types import EvalBundle
    rec = _mkreceipt(tmp_path, "r1", company="ACME", total="10.00")
    pred = _mkpred("r1", company="ACME", total="10.00")
    m = Metrics(
        global_f1=1.0, global_ned=1.0, global_em=1.0,
        per_field_f1={"company": 1.0, "total": 1.0},
        per_field_ned={"company": 1.0, "total": 1.0},
        per_field_em={"company": 1.0, "total": 1.0},
    )
    bundle = EvalBundle(predictions=[pred], receipts=[rec],
                        fields=["company", "total"])
    path = tmp_path / "ext.json"
    n = write_extended_metrics(path, {"donut": (m, bundle)}, n_iter=50, level=0.95)
    out = json.loads(path.read_text())
    assert n > 0
    # Every key is prefixed with 'donut_'.
    assert all(k.startswith("donut_") for k in out)


def test_emit_all_produces_every_expected_sidecar(tmp_path: Path) -> None:
    rec = _mkreceipt(tmp_path, "r1", company="ACME", total="10.00")
    pred = _mkpred("r1", company="ACME", total="10.00")
    m = Metrics(
        global_f1=1.0, global_ned=1.0, global_em=1.0,
        per_field_f1={"company": 1.0, "total": 1.0},
        per_field_ned={"company": 1.0, "total": 1.0},
        per_field_em={"company": 1.0, "total": 1.0},
    )
    counts = emit_all(
        str(tmp_path), ("company", "total"),
        donut_preds=[pred], pipeline_preds=[pred],
        donut_receipts=[rec], pipeline_receipts=[rec],
        donut_metrics=m, pipeline_metrics=m,
        n_iter=50,
    )
    assert counts["donut_preds"] == 1
    assert counts["pipeline_preds"] == 1
    assert counts["extended_keys"] > 0
    # Every sidecar lands under tmp_path/{predictions,metrics}/.
    expected = [
        "predictions/donut_preds.jsonl",
        "predictions/pipeline_preds.jsonl",
        "predictions/donut_errors.jsonl",
        "predictions/pipeline_errors.jsonl",
        "predictions/per_field_errors.jsonl",
        "metrics/extended_metrics.json",
    ]
    for rel in expected:
        assert (tmp_path / rel).is_file(), f"missing: {rel}"


def test_emit_all_skips_systems_with_no_data(tmp_path: Path) -> None:
    """DONUT-only run: pipeline_preds=None should not produce pipeline sidecars."""
    rec = _mkreceipt(tmp_path, "r1", company="ACME")
    pred = _mkpred("r1", company="ACME")
    m = Metrics(
        global_f1=1.0, global_ned=1.0, global_em=1.0,
        per_field_f1={"company": 1.0},
        per_field_ned={"company": 1.0},
        per_field_em={"company": 1.0},
    )
    emit_all(
        str(tmp_path), ("company",),
        donut_preds=[pred], pipeline_preds=None,
        donut_receipts=[rec], pipeline_receipts=[rec],
        donut_metrics=m, pipeline_metrics=None,
        n_iter=50,
    )
    assert (tmp_path / "predictions/donut_preds.jsonl").is_file()
    assert not (tmp_path / "predictions/pipeline_preds.jsonl").is_file()
