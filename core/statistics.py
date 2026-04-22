"""Statistical significance helpers for the paper's confidence intervals.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: produces the 95% bootstrap CI and McNemar p-value that accompany
    the headline F1 numbers in Table I.  NED bucketing drives the
    per-field confusion figure (fig_per_field_confusion).
"""
from __future__ import annotations

import math
import random


def bootstrap_ci(
    per_image_correct: list[bool],
    n_iter: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """Two-sided bootstrap CI on mean per-receipt correctness.

    Defaults match Table I in the paper (``n_iter=1000``, ``ci_level=0.95``),
    but both are plumbed through ``config.bootstrap_n_iter`` and
    ``config.bootstrap_ci_level`` so a reviewer can bump either without
    editing source.
    """
    n = len(per_image_correct)
    if n == 0:
        return (0.0, 0.0)
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    alpha = 1.0 - ci_level
    means: list[float] = []
    for _ in range(n_iter):
        sample = [per_image_correct[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int(math.floor(0.5 * alpha * n_iter)))
    hi_idx = min(n_iter - 1, int(math.ceil((1.0 - 0.5 * alpha) * n_iter)))
    return (means[lo_idx], means[hi_idx])


def paired_bootstrap_delta_ci(
    a_correct: list[bool],
    b_correct: list[bool],
    n_iter: int = 1000,
    ci_level: float = 0.95,
) -> tuple[float, float, float]:
    """Paired bootstrap CI on the per-image correctness delta ``a - b``.

    Returns ``(delta_mean, ci_lo, ci_hi)``.  The paired resample keeps
    the same image in both arms in every iterate, which is the right
    test for ``DONUT`` vs. ``pipeline`` on a shared test set — simple
    unpaired bootstrap wastes the pairing information.
    """
    n = len(a_correct)
    if n == 0 or len(b_correct) != n:
        return (0.0, 0.0, 0.0)
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    alpha = 1.0 - ci_level
    deltas: list[float] = []
    for _ in range(n_iter):
        idxs = [random.randint(0, n - 1) for _ in range(n)]
        a_mean = sum(a_correct[i] for i in idxs) / n
        b_mean = sum(b_correct[i] for i in idxs) / n
        deltas.append(a_mean - b_mean)
    deltas.sort()
    observed = sum(a_correct) / n - sum(b_correct) / n
    lo_idx = max(0, int(math.floor(0.5 * alpha * n_iter)))
    hi_idx = min(n_iter - 1, int(math.ceil((1.0 - 0.5 * alpha) * n_iter)))
    return (observed, deltas[lo_idx], deltas[hi_idx])


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> float:
    """Exact McNemar two-tailed p-value for DONUT vs. pipeline significance."""
    b = sum(1 for a, c in zip(a_correct, b_correct, strict=False) if a and not c)
    c = sum(1 for a, cv in zip(a_correct, b_correct, strict=False) if not a and cv)
    n = b + c
    if n == 0:
        return 1.0
    k_obs = min(b, c)
    # P(X <= k_obs) for X ~ Binom(n, 0.5) via log-sum-exp
    log_half_n = n * math.log(0.5)
    log_binom = 0.0  # log C(n, 0)
    p_one_tail = 0.0
    for k in range(k_obs + 1):
        if k > 0:
            log_binom += math.log(n - k + 1) - math.log(k)
        p_one_tail += math.exp(log_binom + log_half_n)
    return min(1.0, 2.0 * p_one_tail)


def ned_buckets(neds: list[float], _sentinel: object = None) -> dict[str, int]:
    """Bin NED scores into exact/high/mid/low for the per-field confusion fig."""
    counts: dict[str, int] = {"exact": 0, "high": 0, "mid": 0, "low": 0}
    for v in neds:
        if v >= 1.0:
            counts["exact"] += 1
        elif v > 0.7:
            counts["high"] += 1
        elif v > 0.3:
            counts["mid"] += 1
        else:
            counts["low"] += 1
    return counts
