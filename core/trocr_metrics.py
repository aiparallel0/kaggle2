"""TrOCR crop-level diagnostics — CER / WER per field + latency profile.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: reduce a list of ``(gold_text, predicted_text, field_label,
    latency_ms)`` records — produced at pipeline-eval time by
    :mod:`models.trocr_diagnose` — to the headline TrOCR numbers the
    paper's Table VII surfaces: character- and word-error rates (total
    + per-field), plus a latency percentile profile.  The diagnostics
    emitter never calls into torch so this module imports cleanly on
    any CPU-only reviewer checkout.
"""
from __future__ import annotations

from collections.abc import Sequence

from core.metrics import edit_distance
from core.schemas import SCHEMA_VERSIONS, TrocrDiagnostics

# One crop-level record: (gold, pred, field_name, latency_ms).
TrocrRecord = tuple[str, str, str, float]


def _cer(gold: str, pred: str) -> float:
    """Character-error rate — Levenshtein distance / len(gold)."""
    if not gold:
        return 0.0 if not pred else 1.0
    return edit_distance(gold, pred) / len(gold)


def _wer(gold: str, pred: str) -> float:
    """Word-error rate on whitespace-split tokens."""
    g_tokens = gold.split()
    p_tokens = pred.split()
    if not g_tokens:
        return 0.0 if not p_tokens else 1.0
    # Token-level Levenshtein via DP over the two token sequences.
    m, n = len(g_tokens), len(p_tokens)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if g_tokens[i - 1] == p_tokens[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n] / m


def _percentile(values: Sequence[float], q: float) -> float:
    """Inclusive linear-interp percentile (q in [0, 1])."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


def compute_trocr_diagnostics(
    records: Sequence[TrocrRecord],
) -> TrocrDiagnostics:
    """Aggregate per-crop records into the TrOCR diagnostics JSON."""
    if not records:
        return TrocrDiagnostics(schema_version=SCHEMA_VERSIONS["TrocrDiagnostics"])
    cers: list[float] = []
    wers: list[float] = []
    latencies: list[float] = []
    cer_per_field: dict[str, list[float]] = {}
    wer_per_field: dict[str, list[float]] = {}
    total_edits = 0
    total_chars = 0
    for gold, pred, field, lat in records:
        cers.append(_cer(gold, pred))
        wers.append(_wer(gold, pred))
        latencies.append(float(lat))
        cer_per_field.setdefault(field, []).append(cers[-1])
        wer_per_field.setdefault(field, []).append(wers[-1])
        total_edits += edit_distance(gold, pred)
        total_chars += max(1, len(gold))
    return TrocrDiagnostics(
        schema_version=SCHEMA_VERSIONS["TrocrDiagnostics"],
        cer_mean=sum(cers) / len(cers),
        cer_total=total_edits / total_chars,
        wer_mean=sum(wers) / len(wers),
        cer_per_field={k: sum(v) / len(v) for k, v in cer_per_field.items()},
        wer_per_field={k: sum(v) / len(v) for k, v in wer_per_field.items()},
        latency_p50_ms=_percentile(latencies, 0.5),
        latency_p95_ms=_percentile(latencies, 0.95),
        latency_p99_ms=_percentile(latencies, 0.99),
    )
