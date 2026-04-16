"""Unit tests for core.metrics — the metric functions central to F1 claims."""
from __future__ import annotations

from pathlib import Path

from core.metrics import compute_metrics, edit_distance, ned, token_f1
from core.types import Field, Prediction, Receipt

# --- edit_distance ---

def test_edit_distance_identical() -> None:
    assert edit_distance("hello", "hello") == 0


def test_edit_distance_empty_strings() -> None:
    assert edit_distance("", "") == 0


def test_edit_distance_one_empty() -> None:
    assert edit_distance("abc", "") == 3
    assert edit_distance("", "abc") == 3


def test_edit_distance_single_substitution() -> None:
    assert edit_distance("cat", "bat") == 1


def test_edit_distance_insertion_deletion() -> None:
    assert edit_distance("abc", "abcd") == 1
    assert edit_distance("abcd", "abc") == 1


def test_edit_distance_completely_different() -> None:
    assert edit_distance("abc", "xyz") == 3


# --- ned ---

def test_ned_identical() -> None:
    assert ned("hello", "hello") == 1.0


def test_ned_completely_different() -> None:
    assert ned("abc", "xyz") == 0.0


def test_ned_both_empty() -> None:
    assert ned("", "") == 1.0


def test_ned_one_empty() -> None:
    assert ned("abc", "") == 0.0
    assert ned("", "abc") == 0.0


def test_ned_partial_match() -> None:
    # "kitten" → "sitting": edit distance = 3, max len = 7
    result = ned("kitten", "sitting")
    assert abs(result - (1.0 - 3 / 7)) < 1e-6


# --- token_f1 ---

def test_token_f1_identical() -> None:
    assert token_f1("hello world", "hello world") == 1.0


def test_token_f1_both_empty() -> None:
    assert token_f1("", "") == 1.0


def test_token_f1_one_empty() -> None:
    assert token_f1("hello", "") == 0.0
    assert token_f1("", "hello") == 0.0


def test_token_f1_no_overlap() -> None:
    assert token_f1("hello world", "foo bar") == 0.0


def test_token_f1_partial_overlap() -> None:
    # gt = {"hello", "world"}, pred = {"hello", "there"}
    # common = {"hello"}, P = 1/2, R = 1/2, F1 = 0.5
    assert abs(token_f1("hello world", "hello there") - 0.5) < 1e-6


def test_token_f1_superset_prediction() -> None:
    # gt = {"a"}, pred = {"a", "b"}, common = {"a"}, P = 1/2, R = 1/1, F1 = 2/3
    assert abs(token_f1("a", "a b") - 2 / 3) < 1e-6


# --- compute_metrics ---

def _make_receipt(fields: dict[str, str]) -> Receipt:
    return Receipt(
        image_path=Path("dummy.jpg"),
        fields=[Field(name=k, value=v) for k, v in fields.items()],
    )


def _make_prediction(rid: str, fields: dict[str, str]) -> Prediction:
    return Prediction(
        receipt_id=rid,
        fields=[Field(name=k, value=v) for k, v in fields.items()],
    )


def test_compute_metrics_perfect_match() -> None:
    receipts = [_make_receipt({"company": "ACME", "total": "10.00"})]
    preds = [_make_prediction("r1", {"company": "ACME", "total": "10.00"})]
    m = compute_metrics(preds, receipts, ["company", "total"])
    assert m.global_f1 == 1.0
    assert m.global_ned == 1.0
    assert m.global_em == 1.0


def test_compute_metrics_no_match() -> None:
    receipts = [_make_receipt({"company": "ACME"})]
    preds = [_make_prediction("r1", {"company": "ZZZZ"})]
    m = compute_metrics(preds, receipts, ["company"])
    assert m.global_f1 == 0.0
    assert m.global_em == 0.0


def test_compute_metrics_missing_prediction_field() -> None:
    receipts = [_make_receipt({"company": "ACME", "total": "10.00"})]
    preds = [_make_prediction("r1", {"company": "ACME"})]
    m = compute_metrics(preds, receipts, ["company", "total"])
    # company: F1=1.0, total: F1=0.0 → global=0.5
    assert abs(m.global_f1 - 0.5) < 1e-6


def test_compute_metrics_per_field() -> None:
    receipts = [_make_receipt({"company": "A B", "date": "01/01"})]
    preds = [_make_prediction("r1", {"company": "A B", "date": "wrong"})]
    m = compute_metrics(preds, receipts, ["company", "date"])
    assert m.per_field_f1["company"] == 1.0
    assert m.per_field_f1["date"] == 0.0


def test_compute_metrics_case_insensitive() -> None:
    receipts = [_make_receipt({"Company": "ACME Corp"})]
    preds = [_make_prediction("r1", {"company": "acme corp"})]
    m = compute_metrics(preds, receipts, ["company"])
    assert m.global_f1 == 1.0
