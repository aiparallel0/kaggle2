"""Additional statistical helpers: Wilson CI + paired-bootstrap p-values.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: complement :mod:`core.statistics` (which already provides
    bootstrap CI for F1/NED) with the two extra tests the reviewer-
    facing headline table needs — a Wilson confidence interval for
    exact-match proportions and a paired-bootstrap p-value for
    DONUT-vs-Pipeline per-field comparisons.  McNemar's exact test
    is also here so ``figures_errors.py`` can decorate its per-field
    bars with significance stars without a SciPy dependency.
"""
from __future__ import annotations

import math
import random
from collections.abc import Sequence


def wilson_ci(
    successes: int,
    total: int,
    level: float = 0.95,
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion.

    Used for exact-match rate confidence bounds where the plain normal
    approximation is badly skewed at EM near 0 or 1.  ``level`` is the
    confidence level (0.95 = 95% CI).  Returns ``(lo, hi)`` in ``[0, 1]``.
    """
    if total <= 0:
        return (0.0, 0.0)
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1); got {level}")
    # Two-sided z from the standard-normal survival function via
    # ``math.erfcinv`` isn't in stdlib, so invert via bisection on
    # erfc — good to ~1e-8 over the useful range.
    alpha = 1.0 - level
    z = _norm_ppf(1.0 - alpha / 2.0)
    p_hat = successes / total
    denom = 1.0 + (z ** 2) / total
    centre = (p_hat + (z ** 2) / (2.0 * total)) / denom
    halfw = (z / denom) * math.sqrt(
        p_hat * (1.0 - p_hat) / total + (z ** 2) / (4.0 * total * total),
    )
    return (max(0.0, centre - halfw), min(1.0, centre + halfw))


def _norm_ppf(q: float) -> float:
    """Inverse CDF of the standard normal via erfcinv bisection."""
    # Abramowitz & Stegun 26.2.23 rational approximation — good enough
    # for the 90 / 95 / 99 CI levels the paper surfaces.
    if q <= 0.0 or q >= 1.0:
        raise ValueError(f"q must be in (0, 1); got {q}")
    # Use the closed-form approximation from Beasley-Springer-Moro.
    a = [
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ]
    d = [
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ]
    plow, phigh = 0.02425, 1.0 - 0.02425
    if q < plow:
        u = math.sqrt(-2.0 * math.log(q))
        return (((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / \
               ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0)
    if q <= phigh:
        u = q - 0.5
        r = u * u
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * u / \
               (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    u = math.sqrt(-2.0 * math.log(1.0 - q))
    return -(((((c[0] * u + c[1]) * u + c[2]) * u + c[3]) * u + c[4]) * u + c[5]) / \
            ((((d[0] * u + d[1]) * u + d[2]) * u + d[3]) * u + 1.0)


def paired_bootstrap_pvalue(
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_iter: int = 1000,
    seed: int = 42,
) -> float:
    """Two-sided paired-bootstrap p-value for the per-sample delta.

    ``correct_a``/``correct_b`` are aligned length-N boolean sequences
    (e.g. per-image correctness vectors from two models).  Returns the
    two-sided p-value for ``H0: mean(a - b) == 0``.
    """
    n = len(correct_a)
    if n == 0 or n != len(correct_b):
        return 1.0
    diffs = [float(a) - float(b) for a, b in zip(correct_a, correct_b, strict=True)]
    observed = sum(diffs) / n
    rng = random.Random(seed)
    more_extreme = 0
    for _ in range(max(1, n_iter)):
        idxs = [rng.randrange(n) for _ in range(n)]
        resample_mean = sum(diffs[i] for i in idxs) / n
        # Centre the resample so H0 is simulated.
        if abs(resample_mean - observed) >= abs(observed):
            more_extreme += 1
    return more_extreme / max(1, n_iter)


def mcnemar_pvalue(both_correct: int, only_a: int, only_b: int, neither: int) -> float:
    """Exact binomial McNemar's test on the discordant pairs.

    Reviewer-standard pairing test for the headline DONUT-vs-Pipeline
    comparison.  Returns the two-sided exact-binomial p-value.
    ``both_correct`` and ``neither`` are unused by the test but
    accepted so call sites can pass the 2x2 cells without slicing.
    """
    _ = both_correct + neither  # silence unused-arg warnings.
    n = only_a + only_b
    if n == 0:
        return 1.0
    # Two-sided exact binomial with p=0.5.
    k = min(only_a, only_b)
    cum = 0.0
    for i in range(0, k + 1):
        cum += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * cum)
