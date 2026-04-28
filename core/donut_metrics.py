"""DONUT decoder diagnostics — invalid-JSON rate, logprob, token-accuracy.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Role: reduce a list of ``(raw_output, parsed_json_or_none,
    logprob, token_accuracy, gen_length, special_token_hits,
    attn_entropy_per_step, beam_agreed)`` per-sample records — produced
    at eval time by :mod:`models.donut_diagnose` — to the scalars
    surfaced in the paper's Table VIII DONUT diagnostics row.  Never
    raises; missing fields fall back to conservative zeros.
"""
from __future__ import annotations

from collections.abc import Sequence

from core.schemas import SCHEMA_VERSIONS, DonutDiagnostics

# (raw_output, is_valid_json, logprob, token_acc, gen_length,
#  special_token_correct, attn_entropy_mean, beam_agreed).
DonutRecord = tuple[str, bool, float, float, int, bool, float, bool]


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


def compute_donut_diagnostics(
    records: Sequence[DonutRecord],
) -> DonutDiagnostics:
    """Aggregate per-sample DONUT records into the diagnostics JSON."""
    if not records:
        return DonutDiagnostics(schema_version=SCHEMA_VERSIONS["DonutDiagnostics"])
    n = len(records)
    invalid = 0
    logprobs: list[float] = []
    token_accs: list[float] = []
    gen_lens: list[int] = []
    special_hits = 0
    attn_ents: list[float] = []
    beam_agreed = 0
    for _raw, is_valid, lp, tok_acc, gen_len, sp_ok, attn_h, beam_ok in records:
        if not is_valid:
            invalid += 1
        logprobs.append(float(lp))
        token_accs.append(float(tok_acc))
        gen_lens.append(int(gen_len))
        if sp_ok:
            special_hits += 1
        attn_ents.append(float(attn_h))
        if beam_ok:
            beam_agreed += 1
    return DonutDiagnostics(
        schema_version=SCHEMA_VERSIONS["DonutDiagnostics"],
        invalid_json_rate=invalid / n,
        mean_logprob=sum(logprobs) / n,
        token_acc=sum(token_accs) / n,
        attn_entropy_mean=sum(attn_ents) / n,
        gen_len_p50=_percentile([float(x) for x in gen_lens], 0.5),
        gen_len_p95=_percentile([float(x) for x in gen_lens], 0.95),
        special_token_acc=special_hits / n,
        beam_agreement_rate=beam_agreed / n,
    )
