"""Invariant tests on ``extended_metrics.json`` rows (PR #110 follow-up).

For every ``(system, field)`` row written by
:func:`core.extended_metrics.summarise_extended` two invariants must
hold *within a single bundle*:

  * ``min(P, R) - tol ≤ F1 ≤ max(P, R) + tol`` — F1 is the harmonic
    mean of P and R; falling outside ``[min, max]`` is mathematically
    impossible.  When this fired pre-fix it meant the bundle the
    extended-metrics producer received differed from the bundle
    ``compute_metrics`` saw (asymmetric normalisation).

  * ``ci_lo - tol ≤ point ≤ ci_hi + tol`` — the bootstrap brackets
    must contain the point estimate they were generated from.

The synthetic fixture is small (n=4 receipts, 2 fields) so the test
runs in <50 ms with no GPU / no HF Hub — suitable for ``make test``.
"""
from __future__ import annotations

from pathlib import Path

from core.extended_metrics import summarise_extended
from core.metrics import compute_metrics
from core.types import EvalBundle, Field, Prediction, Receipt

_TOL = 1e-6


def _bundle() -> EvalBundle:
    """Return a 4-receipt synthetic bundle with mixed P/R outcomes."""
    rows = [
        ("r1", {"company": "ACME CO", "total": "10.00"},
                {"company": "ACME CO", "total": "10.00"}),
        ("r2", {"company": "BETA SDN BHD", "total": "20.00"},
                {"company": "BETA SDN BHD", "total": "20.00"}),
        ("r3", {"company": "GAMMA STORES", "total": "5.50"},
                {"company": "GAMMA STORES", "total": ""}),
        ("r4", {"company": "DELTA MART", "total": "99.99"},
                {"company": "DELTA", "total": "99.99"}),
    ]
    receipts = [Receipt(image_path=Path(f"/tmp/{rid}.jpg"),
                        fields=[Field(name=k, value=v) for k, v in gt.items()])
                for rid, gt, _ in rows]
    preds = [Prediction(receipt_id=rid,
                        fields=[Field(name=k, value=v) for k, v in pr.items()])
             for rid, _, pr in rows]
    return EvalBundle(predictions=preds, receipts=receipts,
                      fields=["company", "total"])


def test_harmonic_mean_invariant_within_bundle() -> None:
    """For every (system, field) row: ``min(P, R) ≤ F1 ≤ max(P, R)``.

    When this fails it means the producer received a *different*
    bundle than the headline scorer — exactly the PR #110 follow-up
    bug (raw gold vs normalised pred).
    """
    bundle = _bundle()
    metrics = compute_metrics(bundle)
    out = summarise_extended(metrics, bundle, n_iter=50, level=0.95)
    for f in bundle.fields:
        p = float(out[f"precision_{f}"])  # type: ignore[arg-type]
        r = float(out[f"recall_{f}"])  # type: ignore[arg-type]
        f1 = float(metrics.per_field_f1[f])
        assert min(p, r) - _TOL <= f1 <= max(p, r) + _TOL, (
            f"{f}: F1={f1:.4f} not in [{min(p,r):.4f}, {max(p,r):.4f}] "
            f"(P={p:.4f}, R={r:.4f}) — extended-metrics bundle is "
            f"asymmetric with the headline scorer"
        )


def test_bootstrap_brackets_point_estimate() -> None:
    """``ci_lo ≤ <point> ≤ ci_hi`` for every per-field F1 row."""
    bundle = _bundle()
    metrics = compute_metrics(bundle)
    out = summarise_extended(metrics, bundle, n_iter=200, level=0.95)
    for f in bundle.fields:
        lo = float(out[f"f1_{f}_ci_lo"])  # type: ignore[arg-type]
        hi = float(out[f"f1_{f}_ci_hi"])  # type: ignore[arg-type]
        point = float(metrics.per_field_f1[f])
        assert lo - _TOL <= point <= hi + _TOL, (
            f"{f}: F1={point:.4f} outside CI [{lo:.4f}, {hi:.4f}]"
        )
