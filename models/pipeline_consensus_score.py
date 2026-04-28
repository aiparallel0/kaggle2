"""PR-A / T-D2 — scoring half of ``pipeline_consensus`` as its own module.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: namespace home for the per-candidate scoring helpers
    (``_score_total_candidate``, ``_refine_total``, ``_score_money``)
    that the rule-based "consensus" stage applies on top of the
    learned assigner's draft.  The implementation lives in
    :mod:`models.pipeline_consensus`; this module re-exports the
    public symbols so importers can ``from models.pipeline_consensus_score
    import _refine_total`` per the PR-A T-D2 split.

Also hosts the post-PR-A target home for ``models.total_postprocess``
(folded here in T-F1) — currently re-exported so callers do not break.
"""
from __future__ import annotations

# Re-exports — split surface.
from models.pipeline_consensus import (
    _ATTN_BLEND_ALPHA,
    _ATTN_LOG_EPS,
    _TOTAL_OVERRIDE_MARGIN,
    _TOTAL_OVERRIDE_MARGIN_DIFFUSE,
    _is_attn_diffuse,
    _refine_company,
    _refine_date,
    _refine_total,
    _score_money,
)
from models.total_postprocess import extract_total_value

# Public alias requested by the PR-A spec; the refining function is
# still named ``_refine_total`` internally.
_score_total_candidate = _score_money

__all__ = [
    "_ATTN_BLEND_ALPHA",
    "_ATTN_LOG_EPS",
    "_TOTAL_OVERRIDE_MARGIN",
    "_TOTAL_OVERRIDE_MARGIN_DIFFUSE",
    "_is_attn_diffuse",
    "_refine_company",
    "_refine_date",
    "_refine_total",
    "_score_money",
    "_score_total_candidate",
    "extract_total_value",
    "score_address_assembly",
]


def score_address_assembly(
    candidates: list[str], anchor_text: str,
) -> float:
    """PR-C / S0 — address-assembly scorer (token-F1 + line-count + tail).

    Scores how well a multi-line address candidate sequence reconstructs
    the ground truth.  Drops in to :func:`_refine_address` /
    :func:`enforce_address_contiguity` once S0 is enabled in eval.

    Returns a non-negative real where higher is better.  Empty inputs
    score zero.
    """
    if not candidates or not anchor_text:
        return 0.0
    a_tokens = set(anchor_text.lower().split())
    c_tokens = {t for c in candidates for t in c.lower().split()}
    if not a_tokens or not c_tokens:
        return 0.0
    common = a_tokens & c_tokens
    p = len(common) / max(len(c_tokens), 1)
    r = len(common) / max(len(a_tokens), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    # Median-3 line-count prior + postcode-tail bonus + money penalty.
    n_lines = len(candidates)
    line_count_score = max(0.0, 1.0 - abs(n_lines - 3) / 5.0)
    last = candidates[-1].lower()
    postcode_bonus = 0.05 if any(ch.isdigit() for ch in last[-6:]) else 0.0
    money_penalty = 0.10 if any("." in c and c.replace(",", "").replace(".", "")
                                .replace(" ", "").isdigit()
                                for c in candidates) else 0.0
    return f1 + 0.25 * line_count_score + postcode_bonus - money_penalty
