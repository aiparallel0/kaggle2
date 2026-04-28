"""Per-field precision / recall + bootstrap CIs for F1 / NED / EM.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: feed the paper's headline Table I with every column reviewers
    expect — per-field F1 with a 95% CI, per-field EM with a Wilson
    interval, and per-field precision/recall so the reader can see
    *why* one model beats another on e.g. ``total`` (recall-limited)
    vs ``address`` (precision-limited).  Every helper is 2-in/1-out
    and returns a plain ``dict`` so downstream JSON serialisation is
    trivial.
"""
from __future__ import annotations

import random
from collections.abc import Callable, Sequence

from core.extra_stats import wilson_ci
from core.metrics import ned, token_f1
from core.types import EvalBundle, Metrics


def _mean(values: Sequence[float]) -> float:
    """Arithmetic mean — the global-statistic used for token-F1 / NED / EM.

    The headline ``donut_f1_<field>`` / ``pipeline_f1_<field>`` numbers
    are themselves arithmetic means over per-image token-F1
    (see ``core.metrics.compute_metrics``).  Bootstrapping the same
    statistic ensures the returned ``(lo, hi)`` brackets the point
    estimate — see :func:`_bootstrap_field`.
    """
    return sum(values) / len(values) if values else 0.0


def _pairs(bundle: EvalBundle) -> list[tuple[str, dict[str, str], dict[str, str]]]:
    """Flatten an :class:`EvalBundle` into ``(image_id, gt, pred)`` triples."""
    out: list[tuple[str, dict[str, str], dict[str, str]]] = []
    for pred, rec in zip(bundle.predictions, bundle.receipts, strict=True):
        gt = {fld.name.lower(): fld.value.lower() for fld in rec.fields}
        pr = {fld.name.lower(): fld.value.lower() for fld in pred.fields}
        out.append((rec.image_path.stem, gt, pr))
    return out


def _token_pr(g: str, p: str) -> tuple[float, float]:
    """Whitespace-token precision + recall (2-in/1-out returns a tuple)."""
    ta, tb = set(g.split()), set(p.split())
    if not ta and not tb:
        return (1.0, 1.0)
    if not ta:
        return (0.0, 1.0)
    if not tb:
        return (1.0, 0.0)
    common = ta & tb
    return (len(common) / len(tb), len(common) / len(ta))


def per_field_precision(bundle: EvalBundle) -> dict[str, float]:
    """Mean token-precision per field across the eval bundle."""
    acc: dict[str, list[float]] = {f: [] for f in bundle.fields}
    for _, gt, pr in _pairs(bundle):
        for f in bundle.fields:
            p, _r = _token_pr(gt.get(f, ""), pr.get(f, ""))
            acc[f].append(p)
    return {f: sum(v) / len(v) if v else 0.0 for f, v in acc.items()}


def per_field_recall(bundle: EvalBundle) -> dict[str, float]:
    """Mean token-recall per field across the eval bundle."""
    acc: dict[str, list[float]] = {f: [] for f in bundle.fields}
    for _, gt, pr in _pairs(bundle):
        for f in bundle.fields:
            _p, r = _token_pr(gt.get(f, ""), pr.get(f, ""))
            acc[f].append(r)
    return {f: sum(v) / len(v) if v else 0.0 for f, v in acc.items()}


def _bootstrap_field(
    values: Sequence[float], n_iter: int, level: float, seed: int,
    statistic_fn: Callable[[Sequence[float]], float] = _mean,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the *global statistic* over ``values``.

    Resamples the per-image vector ``values`` with replacement ``n_iter``
    times and records ``statistic_fn(resample)`` on each draw.  The
    returned ``(lo, hi)`` is the level-``level`` percentile interval of
    that bootstrap distribution — i.e. the CI of the **global token-F1
    estimator** itself, *not* of the per-image-mean which would be a
    different (narrower) statistic.

    Defaulting ``statistic_fn`` to :func:`_mean` keeps the math
    equivalent to the previous implementation (``donut_f1_<field>`` is
    a mean-of-per-image-F1) while making the contract explicit so
    :func:`assert_ci_bounds_valid` can verify
    ``ci_lo <= <sys>_f1_<field> <= ci_hi`` on every reference run.
    """
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    stats: list[float] = []
    for _ in range(max(1, n_iter)):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        stats.append(statistic_fn(resample))
    stats.sort()
    alpha = 1.0 - level
    lo_idx = int(alpha / 2.0 * len(stats))
    hi_idx = int((1.0 - alpha / 2.0) * len(stats)) - 1
    return (stats[max(0, lo_idx)], stats[max(0, min(hi_idx, len(stats) - 1))])


def per_field_bootstrap_ci(
    bundle: EvalBundle, n_iter: int = 1000, level: float = 0.95, seed: int = 42,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Bootstrap CIs for per-field F1 / NED / EM.

    Returns a nested dict keyed ``{"f1": {field: (lo, hi)}, "ned": ...,
    "em": ...}`` so downstream serialisation is a single nested loop.
    """
    f1s: dict[str, list[float]] = {f: [] for f in bundle.fields}
    neds: dict[str, list[float]] = {f: [] for f in bundle.fields}
    ems: dict[str, list[float]] = {f: [] for f in bundle.fields}
    for _, gt, pr in _pairs(bundle):
        for f in bundle.fields:
            g = gt.get(f, "")
            p = pr.get(f, "")
            f1s[f].append(token_f1(g, p))
            neds[f].append(ned(g, p))
            ems[f].append(1.0 if g == p else 0.0)
    return {
        "f1": {f: _bootstrap_field(v, n_iter, level, seed) for f, v in f1s.items()},
        "ned": {f: _bootstrap_field(v, n_iter, level, seed) for f, v in neds.items()},
        "em": {f: _bootstrap_field(v, n_iter, level, seed) for f, v in ems.items()},
    }


def per_field_em_wilson(
    bundle: EvalBundle, level: float = 0.95,
) -> dict[str, tuple[float, float]]:
    """Wilson CI for the per-field exact-match proportion."""
    counts: dict[str, tuple[int, int]] = {f: (0, 0) for f in bundle.fields}
    for _, gt, pr in _pairs(bundle):
        for f in bundle.fields:
            s, n = counts[f]
            counts[f] = (s + int(gt.get(f, "") == pr.get(f, "")), n + 1)
    return {f: wilson_ci(s, n, level) for f, (s, n) in counts.items()}


def summarise_extended(
    metrics: Metrics, bundle: EvalBundle, n_iter: int = 1000, level: float = 0.95,
) -> dict[str, object]:
    """Flat dict the paper injector can consume — every key reviewer-ready.

    The ``f1_<field>_ci_lo`` / ``f1_<field>_ci_hi`` pair is the
    percentile bootstrap CI of the **global token-F1 estimator**
    (:func:`_bootstrap_field` resamples the per-image vector and
    recomputes the same arithmetic-mean statistic that
    ``core.metrics.compute_metrics`` writes to
    ``Metrics.per_field_f1[field]``).  This is the matched-statistic
    estimator-CI invariant that :func:`report.missing.assert_ci_bounds_valid`
    relies on: the bare point estimate written by
    ``report.combine.build_combined`` (``<sys>_f1_<field>``) is
    bracketed by ``[ci_lo, ci_hi]`` because both come from the same
    statistic over the same sample.
    """
    p = per_field_precision(bundle)
    r = per_field_recall(bundle)
    cis = per_field_bootstrap_ci(bundle, n_iter=n_iter, level=level)
    em_ci = per_field_em_wilson(bundle, level=level)
    out: dict[str, object] = {}
    for f in bundle.fields:
        out[f"precision_{f}"] = p[f]
        out[f"recall_{f}"] = r[f]
        # Per-field point estimates surface ``\VAR{<sys>_ned_<field>}`` and
        # ``\VAR{<sys>_em_<field>}`` in the paper's appendix per-field
        # table; without them only the CI bounds would be available and
        # the cell would render bound-only, breaking bus-accuracy.
        out[f"ned_{f}"] = metrics.per_field_ned[f]
        out[f"em_{f}"] = metrics.per_field_em[f]
        out[f"f1_{f}_ci_lo"] = cis["f1"][f][0]
        out[f"f1_{f}_ci_hi"] = cis["f1"][f][1]
        out[f"ned_{f}_ci_lo"] = cis["ned"][f][0]
        out[f"ned_{f}_ci_hi"] = cis["ned"][f][1]
        out[f"em_{f}_ci_lo"] = em_ci[f][0]
        out[f"em_{f}_ci_hi"] = em_ci[f][1]
    out["f1_macro"] = metrics.global_f1
    out["ned_macro"] = metrics.global_ned
    out["em_macro"] = metrics.global_em
    return out
