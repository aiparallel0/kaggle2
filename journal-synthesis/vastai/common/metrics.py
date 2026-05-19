"""Real metric implementations for the vast.ai journal experiments.

Ported VERBATIM (math unchanged) from
journal-synthesis/experiments/run_analysis.py so numbers computed off-box
are directly comparable with the stdlib E1-E4 run, plus the additional
estimators the blocked experiments need (bootstrap CI, Spearman rho,
log2 variance ratio for the Axis-B beam-margin compression statistic).

Nothing here fabricates a value. Every function is a pure transform of
its inputs; callers must feed it real model output.
"""
from __future__ import annotations

import math
import random
from typing import List, Sequence, Tuple

DEFAULT_SEED = 12345


def median(xs: Sequence[float]):
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return None
    if n % 2:
        return s[n // 2]
    return 0.5 * (s[n // 2 - 1] + s[n // 2])


def phi_mcc(a: Sequence[int], b: Sequence[int]) -> Tuple[float, Tuple[int, int, int, int]]:
    """Matthews / phi correlation for two binary lists (1 == event).

    Identical implementation to run_analysis.py:phi_mcc.
    """
    n11 = sum(1 for x, y in zip(a, b) if x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n00 = sum(1 for x, y in zip(a, b) if not x and not y)
    num = n11 * n00 - n10 * n01
    den = ((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)) ** 0.5
    phi = 0.0 if den == 0 else num / den
    return phi, (n11, n10, n01, n00)


def perm_p(a: Sequence[int], b: Sequence[int], iters: int = 20000,
           seed: int = DEFAULT_SEED) -> float:
    """Two-sided permutation p-value on |phi| by shuffling b.

    Identical to run_analysis.py:perm_p (seed default 12345).
    """
    obs = abs(phi_mcc(a, b)[0])
    rng = random.Random(seed)
    bb = list(b)
    ge = 0
    for _ in range(iters):
        rng.shuffle(bb)
        if abs(phi_mcc(a, bb)[0]) >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (iters + 1)


def wilson(k: int, n: int, z: float = 1.96):
    """Wilson 95% binomial CI. Identical to run_analysis.py:wilson and to
    triology core.stats.wilson_ci convention (Paper 1 Table II)."""
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return ((c - h) / d, (c + h) / d)


def bootstrap_ci(values: Sequence[float], stat=None, iters: int = 10000,
                 alpha: float = 0.05, seed: int = DEFAULT_SEED):
    """Percentile bootstrap CI for an arbitrary statistic over `values`.

    Default statistic is the mean. Deterministic given `seed`. Used for
    paired-bootstrap delta CIs (E5 head-to-head, E9 bake-off) by passing
    paired difference values.
    """
    if not values:
        return (None, None, None)
    if stat is None:
        stat = lambda xs: sum(xs) / len(xs)
    rng = random.Random(seed)
    n = len(values)
    obs = stat(values)
    boots = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(stat(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * iters)]
    hi = boots[int((1 - alpha / 2) * iters) - 1]
    return (obs, lo, hi)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rank correlation (no scipy dependency). NaN-safe: returns
    0.0 for degenerate (constant) input."""
    n = len(x)
    if n < 2 or len(y) != n:
        return 0.0

    def ranks(v):
        order = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(x), ranks(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx == 0 or vy == 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def auroc(pos: Sequence[float], neg: Sequence[float]):
    """Rank-based AUROC = Mann-Whitney U / (n_pos * n_neg), ties at 0.5.

    `pos` = scores for the class labelled 1 (here: shifted corpus),
    `neg` = scores for the class labelled 0 (here: in-distribution).
    Returns the probability a random positive outranks a random negative
    (0.5 = no separation, 1.0 = positives always rank higher, 0.0 =
    perfectly reversed). Pure stdlib, exact tie handling (each tied
    pair contributes 0.5). Returns None if either class is empty.

    Computed via the rank-sum identity
        U = R_pos - n_pos*(n_pos+1)/2
    on midranks (ties averaged), which is exactly equivalent to the
    O(n_pos*n_neg) pairwise count with 0.5 per tie, but O(n log n).
    """
    npos, nneg = len(pos), len(neg)
    if npos == 0 or nneg == 0:
        return None
    combined = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    combined.sort(key=lambda t: t[0])
    n = len(combined)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based midrank
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    r_pos = sum(ranks[k] for k in range(n) if combined[k][1] == 1)
    u = r_pos - npos * (npos + 1) / 2.0
    return u / (npos * nneg)


def variance(xs: Sequence[float]) -> float:
    """Population-free sample variance (n-1), matches statistics.variance."""
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return sum((v - m) ** 2 for v in xs) / (n - 1)


def variance_ratio_log2(in_dist: Sequence[float],
                        shifted: Sequence[float]) -> float:
    """Axis-B compression statistic: log2( Var[in-dist margins] /
    Var[shifted margins] ). Matches arith-gating BM2_compression.py
    (log2 C = log2(var(a)/var(b))). Positive => margins COMPRESS under
    shift (the Paper 2 mechanistic prediction)."""
    va = variance(in_dist)
    vb = variance(shifted)
    if va <= 0 or vb <= 0:
        return float("nan")
    return math.log2(va / vb)
