"""Statistical tests for per-image evaluation.

Pure functions: bootstrap CI, McNemar exact test, NED bucket histogram.

Public API (2-in / 1-out):
  bootstrap_ci(per_image_correct, n_iter) -> (lo, hi)
  mcnemar(a_correct, b_correct) -> p_value
  ned_buckets(neds, _sentinel) -> dict
"""
from __future__ import annotations

import math
import random


def bootstrap_ci(
    per_image_correct: list[bool], n_iter: int = 1000
) -> tuple[float, float]:
    """95 % bootstrap CI on mean correctness by image-level resampling.

    Args:
        per_image_correct: Binary per-image correctness vector.
        n_iter: Number of bootstrap resamples (default 1 000).

    Returns:
        (lo, hi) — 2.5th and 97.5th percentile of the bootstrap distribution.
    """
    n = len(per_image_correct)
    if n == 0:
        return (0.0, 0.0)
    means: list[float] = []
    for _ in range(n_iter):
        sample = [per_image_correct[random.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo_idx = max(0, int(math.floor(0.025 * n_iter)))
    hi_idx = min(n_iter - 1, int(math.ceil(0.975 * n_iter)))
    return (means[lo_idx], means[hi_idx])


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> float:
    """Exact McNemar p-value for paired per-image correctness vectors.

    Uses the exact binomial test under H0: p = 0.5.  Two-tailed.

    Args:
        a_correct: System-A binary correctness (one entry per test image).
        b_correct: System-B binary correctness (same length as a_correct).

    Returns:
        Two-tailed p-value in [0, 1].
    """
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
    """Bin NED scores into 4 buckets used by the per-field confusion figure.

    Buckets follow the NED convention where 1.0 = identical and 0.0 = entirely
    different:  ``exact`` (NED = 1), ``high`` (0.7 < NED < 1), ``mid``
    (0.3 < NED <= 0.7), ``low`` (NED <= 0.3, including 0).

    Args:
        neds: List of NED values in [0, 1].
        _sentinel: Unused; present to satisfy the 2-in / 1-out contract shape.

    Returns:
        Dict with keys ``exact``, ``high``, ``mid``, ``low``.
    """
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
