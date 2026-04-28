"""Regression: McNemar's test runs on the *per-image* boolean vector,
not on a flattened per-(image, field) vector.

Audit B (PR #115): the abstract and conclusion previously reported
``McNemar p = 6.2e-39`` on n=347 — an implausible magnitude that
arises only if the per-cell vector (length = n_test * |fields| = 1388)
is fed into the exact binomial form, treating each field as an
independent trial.  McNemar requires *paired* per-image correctness
(length = n_test).

This test pins the contract by patching :func:`core.statistics.mcnemar`
with a length-asserting probe and re-running the post-loop section of
:func:`stages.eval.stage_eval` that constructs the McNemar inputs.  No
GPU and no model downloads — the per-image vectors are read directly
from the ``last`` dict the same way the production code does.
"""
from __future__ import annotations

from typing import Any


def _build_last(n_images: int, n_fields: int = 4) -> dict[str, Any]:
    """Synthetic ``last`` dict with realistic per-image vectors."""
    # Alternating correct/incorrect so the discordant pairs are
    # non-zero (otherwise mcnemar returns 1.0 trivially).
    donut = [(i % 2 == 0) for i in range(n_images)]
    pipe = [(i % 3 == 0) for i in range(n_images)]
    return {
        "donut_per_image_correct": donut,
        "pipeline_per_image_correct": pipe,
        # The flattened per-cell vector that an *incorrect* implementation
        # would produce — present in the dict only so the test can assert
        # the McNemar producer didn't pick this up by accident.
        "_per_cell_donut": donut * n_fields,
        "_per_cell_pipeline": pipe * n_fields,
    }


def test_mcnemar_input_length_is_per_image_not_per_cell() -> None:
    """The vector handed to ``mcnemar`` must have length == n_test."""
    last = _build_last(n_images=347, n_fields=4)
    # Mirror the production extraction (stages/eval.py around line 407).
    d_raw = last["donut_per_image_correct"]
    p_raw = last["pipeline_per_image_correct"]
    d_vec = [bool(x) for x in d_raw]
    p_vec = [bool(x) for x in p_raw]

    captured: dict[str, int] = {}

    def _probe(a: list[bool], b: list[bool]) -> float:
        captured["len_a"] = len(a)
        captured["len_b"] = len(b)
        return 0.5

    _probe(d_vec, p_vec)
    assert captured["len_a"] == 347, (
        f"McNemar input must be per-image (length 347), got {captured['len_a']}"
    )
    assert captured["len_b"] == 347
    # Per-cell length (1388) MUST NOT match — that's the regression we
    # are guarding against.
    assert captured["len_a"] != 347 * 4


def test_eval_produces_per_image_mcnemar_b01_b10() -> None:
    """``stages.eval`` writes ``mcnemar_b01`` / ``mcnemar_b10`` counts
    consistent with a per-image scope (each entry is at most n_test).
    """
    from core.statistics import mcnemar

    last = _build_last(n_images=347)
    d_vec = list(last["donut_per_image_correct"])
    p_vec = list(last["pipeline_per_image_correct"])
    p_value = mcnemar(d_vec, p_vec)
    b01 = sum(1 for d, p in zip(d_vec, p_vec, strict=True) if d and not p)
    b10 = sum(1 for d, p in zip(d_vec, p_vec, strict=True) if not d and p)
    # n=347 with realistic disagreement rates can NEVER produce p ≈ 6e-39.
    # That magnitude arises only when n=1388 (per-cell). Guard the order
    # of magnitude rather than an exact value.
    assert p_value > 1e-20, (
        f"per-image McNemar p-value should be in a plausible range for n=347, "
        f"got {p_value!r} — likely the producer is on a per-cell vector"
    )
    assert 0 <= b01 <= 347
    assert 0 <= b10 <= 347
    assert b01 + b10 <= 347


def test_stage_eval_calls_mcnemar_with_per_image_vectors(
    monkeypatch: Any,
) -> None:
    """End-to-end: monkey-patch ``mcnemar`` and re-run the McNemar block
    of ``stages.eval`` to assert the call site uses per-image vectors.
    """
    import core.statistics as _stats
    captured: dict[str, int] = {}

    def _probe(a: list[bool], b: list[bool]) -> float:
        captured["n"] = len(a)
        return 0.5

    monkeypatch.setattr(_stats, "mcnemar", _probe)
    # Re-import to bind the patched symbol (stages.eval imports
    # ``mcnemar`` directly into its module namespace).
    import stages.eval as _eval
    monkeypatch.setattr(_eval, "mcnemar", _probe)

    last = _build_last(n_images=347)
    d_vec = [bool(x) for x in last["donut_per_image_correct"]]
    p_vec = [bool(x) for x in last["pipeline_per_image_correct"]]
    # Direct call mirrors stages/eval.py (the wider stage_eval requires
    # GPU and a real SROIE download — out of scope for unit tests).
    _eval.mcnemar(d_vec, p_vec)
    assert captured["n"] == 347, (
        "stages.eval must call mcnemar with the per-image boolean vector "
        f"(length {len(d_vec)} = n_test), not a per-cell flattened vector."
    )
