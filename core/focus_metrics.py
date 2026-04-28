"""Attention-assigner diagnostics — entropy, calibration, top-k accuracy.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: turn a list of ``(field, attention_distribution, is_correct)``
    records — produced at pipeline-eval time by
    :mod:`models.assigner_diagnose` — into the scalars surfaced in
    the paper's Table VIII: per-field attention entropy, peak
    sharpness (max − mean), expected calibration error (ECE), Brier
    score, top-1/3/5 accuracy, and per-level accuracy (L1 vs L2 in
    the hierarchical assigner).  Torch-free so CPU-only reviewer
    checkouts can import and smoke-test the emitter.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

from core.schemas import SCHEMA_VERSIONS, AssignerDiagnostics

# (field, attention_softmax, is_correct, top_choice_idx, true_idx).
AssignerRecord = tuple[str, Sequence[float], bool, int, int]


def _entropy(dist: Sequence[float]) -> float:
    """Shannon entropy in bits; zero-probability masses dropped."""
    total = sum(max(0.0, float(p)) for p in dist)
    if total <= 0.0:
        return 0.0
    h = 0.0
    for p in dist:
        q = max(0.0, float(p)) / total
        if q > 0.0:
            h -= q * math.log2(q)
    return h


def _peak_sharpness(dist: Sequence[float]) -> float:
    """``max(p) − mean(p)`` — cheap proxy for how peaked the attention is."""
    if not dist:
        return 0.0
    vals = [max(0.0, float(p)) for p in dist]
    return max(vals) - (sum(vals) / len(vals))


def _ece(
    confidences: Sequence[float], correct: Sequence[bool], n_bins: int = 10,
) -> tuple[float, float]:
    """Expected + maximum calibration error with ``n_bins`` equal-width bins."""
    if not confidences or len(confidences) != len(correct):
        return (0.0, 0.0)
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for c, k in zip(confidences, correct, strict=True):
        idx = min(n_bins - 1, int(c * n_bins))
        bins[idx].append((c, k))
    n = len(confidences)
    ece_acc = 0.0
    mce = 0.0
    for b in bins:
        if not b:
            continue
        avg_conf = sum(c for c, _ in b) / len(b)
        acc = sum(1 for _, k in b if k) / len(b)
        gap = abs(avg_conf - acc)
        ece_acc += (len(b) / n) * gap
        mce = max(mce, gap)
    return (ece_acc, mce)


def _brier(confidences: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the confidence against the 0/1 correctness."""
    if not confidences:
        return 0.0
    s = 0.0
    for c, k in zip(confidences, correct, strict=True):
        target = 1.0 if k else 0.0
        s += (c - target) ** 2
    return s / len(confidences)


def _topk_acc(
    dists: Sequence[Sequence[float]], trues: Sequence[int], k: int,
) -> float:
    """Top-k accuracy — 1 if the true index is in the top-k of the distribution."""
    if not dists:
        return 0.0
    hits = 0
    for dist, t in zip(dists, trues, strict=True):
        ranked = sorted(range(len(dist)), key=lambda i: -float(dist[i]))
        if t in ranked[:k]:
            hits += 1
    return hits / len(dists)


def compute_assigner_diagnostics(
    records: Sequence[AssignerRecord],
    level1_correct: Sequence[bool] = (),
    level2_correct: Sequence[bool] = (),
    prior_posterior_kl: float = 0.0,
) -> AssignerDiagnostics:
    """Reduce assigner records to the paper's diagnostics dict."""
    if not records:
        return AssignerDiagnostics(schema_version=SCHEMA_VERSIONS["AssignerDiagnostics"])
    entropy_per_field: dict[str, list[float]] = {}
    sharpness_per_field: dict[str, list[float]] = {}
    dists: list[Sequence[float]] = []
    trues: list[int] = []
    confs: list[float] = []
    correct: list[bool] = []
    for field, dist, is_corr, top_idx, true_idx in records:
        entropy_per_field.setdefault(field, []).append(_entropy(dist))
        sharpness_per_field.setdefault(field, []).append(_peak_sharpness(dist))
        dists.append(dist)
        trues.append(true_idx)
        confs.append(float(dist[top_idx]) if 0 <= top_idx < len(dist) else 0.0)
        correct.append(bool(is_corr))
    ece, mce = _ece(confs, correct)
    return AssignerDiagnostics(
        schema_version=SCHEMA_VERSIONS["AssignerDiagnostics"],
        entropy_per_field={k: sum(v) / len(v) for k, v in entropy_per_field.items()},
        attention_peak_sharpness={
            k: sum(v) / len(v) for k, v in sharpness_per_field.items()
        },
        ece=ece, mce=mce, brier=_brier(confs, correct),
        top1_acc=_topk_acc(dists, trues, 1),
        top3_acc=_topk_acc(dists, trues, 3),
        top5_acc=_topk_acc(dists, trues, 5),
        level1_acc=(
            sum(1 for x in level1_correct if x) / len(level1_correct)
            if level1_correct else 0.0
        ),
        level2_acc=(
            sum(1 for x in level2_correct if x) / len(level2_correct)
            if level2_correct else 0.0
        ),
        prior_posterior_kl=float(prior_posterior_kl),
    )
