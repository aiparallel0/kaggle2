"""Canonical F1 / NED / EM computation over an :class:`EvalBundle`.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: single source of truth for the three headline quantities
    reported in the paper — token-F1, Normalised Edit Distance (NED),
    and Exact Match (EM) — together with the Levenshtein edit-distance
    primitive they share.  All evaluation paths (DONUT, the learned
    pipeline arm, the rule-based baseline, and the gold-OCR ablation)
    route through :func:`compute_metrics` so cross-system numbers are
    directly comparable and traceable to one implementation.

    For backwards compatibility the module also re-exports the
    statistical helpers from :mod:`core.statistics` (``bootstrap_ci``,
    ``mcnemar``, ``ned_buckets``) and the :class:`CombinedMetrics`
    schema from :mod:`core.combined_metrics`, so callers written
    before the 166-LOC split continue to import from this module.
"""
from __future__ import annotations

from core.combined_metrics import CombinedMetrics as CombinedMetrics  # noqa: PLC0414
from core.statistics import bootstrap_ci as bootstrap_ci  # noqa: PLC0414
from core.statistics import mcnemar as mcnemar  # noqa: PLC0414
from core.statistics import ned_buckets as ned_buckets  # noqa: PLC0414
from core.types import EvalBundle, Metrics


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings.

    Classical two-row dynamic-programming formulation: O(mn) time,
    O(min(m, n)) memory.  Returned unit is the count of insertions,
    deletions, and substitutions that turn *a* into *b*; used both
    directly (for NED) and as an input to the bucketised NED analysis
    reported in the paper's robustness sub-section.
    """
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
    """Normalised Edit Distance as a similarity score in ``[0, 1]``.

    Defined as ``1 − edit_distance(a, b) / max(|a|, |b|)`` so that
    ``1.0`` denotes identical strings and ``0.0`` denotes maximally
    different strings of equal length.  Two empty strings return
    ``1.0`` (the convention used by the SROIE leaderboard).
    """
    if not a and not b:
        return 1.0
    dist = edit_distance(a, b)
    return 1.0 - dist / max(len(a), len(b))


def token_f1(a: str, b: str) -> float:
    """Whitespace-token F1 between ground-truth *a* and prediction *b*.

    Tokens are compared as a *set* so repeated tokens inside a field
    value do not artificially inflate recall.  Two empty strings
    return ``1.0``; any asymmetric empty/non-empty pair returns
    ``0.0``.  This matches the public SROIE evaluation script.
    """
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
    """Reduce an :class:`EvalBundle` to per-field and global F1/NED/EM.

    Each ``(receipt, prediction)`` pair is reduced field-by-field
    after case-folding the expected and predicted values.  Global
    scores are the unweighted mean over the four SROIE fields, which
    matches the reference implementation and keeps the four fields
    equally weighted regardless of their relative prevalence.

    ``per_image_correct`` is set to ``True`` for a receipt only when
    every field is an exact match — the binary signal used by the
    McNemar test in the paper.  ``per_image_f1`` is the mean per-field
    token-F1 for the same receipt and is the right input for the
    bootstrap CI accompanying the headline F1 number (the all-fields-EM
    vector is degenerate — and silently zero — whenever no receipt has
    every field correct simultaneously, which is the failure mode that
    produced the zero-width ``pipeline_bootstrap_ci_*`` keys in
    earlier runs).
    """
    pf1: dict[str, list[float]] = {f: [] for f in bundle.fields}
    pned: dict[str, list[float]] = {f: [] for f in bundle.fields}
    pem: dict[str, list[float]] = {f: [] for f in bundle.fields}
    per_image_ok: list[bool] = []
    per_image_macro_f1: list[float] = []
    for pred, rec in zip(bundle.predictions, bundle.receipts, strict=True):
        gt = {fld.name.lower(): fld.value.lower() for fld in rec.fields}
        pr = {fld.name.lower(): fld.value.lower() for fld in pred.fields}
        all_fields_match = True
        per_field_f1_for_image: list[float] = []
        for f in bundle.fields:
            g = gt.get(f, "")
            p = pr.get(f, "")
            match = (g == p)
            pem[f].append(1.0 if match else 0.0)
            pned[f].append(ned(g, p))
            f1_value = token_f1(g, p)
            pf1[f].append(f1_value)
            per_field_f1_for_image.append(f1_value)
            if not match:
                all_fields_match = False
        per_image_ok.append(all_fields_match)
        per_image_macro_f1.append(
            sum(per_field_f1_for_image) / len(per_field_f1_for_image)
            if per_field_f1_for_image else 0.0,
        )
    per_f1 = {f: sum(v) / len(v) for f, v in pf1.items() if v}
    per_ned = {f: sum(v) / len(v) for f, v in pned.items() if v}
    per_em = {f: sum(v) / len(v) for f, v in pem.items() if v}
    g_f1 = sum(per_f1.values()) / len(per_f1) if per_f1 else 0.0
    g_ned = sum(per_ned.values()) / len(per_ned) if per_ned else 0.0
    g_em = sum(per_em.values()) / len(per_em) if per_em else 0.0
    return Metrics(
        global_f1=g_f1, global_ned=g_ned, global_em=g_em,
        per_field_f1=per_f1, per_field_ned=per_ned, per_field_em=per_em,
        per_image_correct=per_image_ok,
        per_image_f1=per_image_macro_f1,
    )
