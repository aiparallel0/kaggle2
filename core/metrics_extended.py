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
from collections.abc import Sequence

from core.metrics import ned, token_f1
from core.stats_extra import wilson_ci
from core.types import EvalBundle, Metrics


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
) -> tuple[float, float]:
    """Percentile bootstrap CI for a single per-field scalar vector."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(max(1, n_iter)):
        idxs = [rng.randrange(n) for _ in range(n)]
        means.append(sum(values[i] for i in idxs) / n)
    means.sort()
    alpha = 1.0 - level
    lo_idx = int(alpha / 2.0 * len(means))
    hi_idx = int((1.0 - alpha / 2.0) * len(means)) - 1
    return (means[max(0, lo_idx)], means[max(0, min(hi_idx, len(means) - 1))])


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
    """Flat dict the paper injector can consume — every key reviewer-ready."""
    p = per_field_precision(bundle)
    r = per_field_recall(bundle)
    cis = per_field_bootstrap_ci(bundle, n_iter=n_iter, level=level)
    em_ci = per_field_em_wilson(bundle, level=level)
    out: dict[str, object] = {}
    for f in bundle.fields:
        out[f"precision_{f}"] = p[f]
        out[f"recall_{f}"] = r[f]
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
