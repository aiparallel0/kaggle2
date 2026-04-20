"""Shared metric computation: token-F1, NED, EM — used by both eval modules."""
from __future__ import annotations

from core.types import EvalBundle, Metrics


def edit_distance(a: str, b: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def ned(a: str, b: str) -> float:
    """Normalised Edit Distance: 1.0 = identical, 0.0 = completely different."""
    if not a and not b:
        return 1.0
    dist = edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


def token_f1(a: str, b: str) -> float:
    """Token-level F1 between ground-truth *a* and prediction *b*."""
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    common = ta & tb
    p = len(common) / len(tb)
    r = len(common) / len(ta)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def compute_metrics(bundle: EvalBundle) -> Metrics:
    """Compute per-field and global F1 / NED / EM from an :class:`EvalBundle`."""
    pf1: dict[str, list[float]] = {f: [] for f in bundle.fields}
    pned: dict[str, list[float]] = {f: [] for f in bundle.fields}
    pem: dict[str, list[float]] = {f: [] for f in bundle.fields}
    for pred, rec in zip(bundle.predictions, bundle.receipts, strict=True):
        gt = {fld.name.lower(): fld.value.lower() for fld in rec.fields}
        pr = {fld.name.lower(): fld.value.lower() for fld in pred.fields}
        for f in bundle.fields:
            g = gt.get(f, "")
            p = pr.get(f, "")
            pem[f].append(1.0 if g == p else 0.0)
            pned[f].append(ned(g, p))
            pf1[f].append(token_f1(g, p))
    per_f1 = {f: sum(v) / len(v) for f, v in pf1.items() if v}
    per_ned = {f: sum(v) / len(v) for f, v in pned.items() if v}
    per_em = {f: sum(v) / len(v) for f, v in pem.items() if v}
    g_f1 = sum(per_f1.values()) / len(per_f1) if per_f1 else 0.0
    g_ned = sum(per_ned.values()) / len(per_ned) if per_ned else 0.0
    g_em = sum(per_em.values()) / len(per_em) if per_em else 0.0
    return Metrics(
        global_f1=g_f1, global_ned=g_ned, global_em=g_em,
        per_field_f1=per_f1, per_field_ned=per_ned, per_field_em=per_em,
    )
