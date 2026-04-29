"""Field assignment via the learned AttentionAssigner's cross-attention.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: performs inference-time field assignment by picking regions whose
    attention exceeds half of max for multi-line fields (address), then
    postprocessing date/total through regex to match SROIE GT format.

PR-A / T-D1 — public dataclasses :class:`AssignerPolicy` and
:class:`AssignerInputs` collapse the previous 7-kwarg surface of
``_assign_learned_with_attn`` into the 2-in/1-out contract mandated by
``AGENTS.md``.  The legacy 7-kwarg signature is preserved for one PR
as a thin shim so existing callers do not break.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from core.types import ZoneConfig
from models.consensus import (
    _is_addr_boundary as _consensus_is_addr_boundary,
)
from models.consensus import enforce_address_contiguity
from models.date_post import fallback_from_ocr_lines, is_plausible
from models.focus_addr_penalty import shrink_addr_span
from models.focus_inference import (
    N_TEXT_PRIORS,
    N_TEXT_PRIORS_V2,
    N_TEXT_PRIORS_V3,
    N_TEXT_PRIORS_V4,
    AttentionAssigner,
    arithmetic_witnesses_v4,
    text_priors,
    text_priors_v2,
    text_priors_v3,
    text_priors_v4,
)
from models.focus_priors import _MONEY_RE as _PRIORS_MONEY_RE
from models.focus_priors import (
    V4_IS_COMPANY_BOILERPLATE_IDX,
    V4_WITNESS_IDX,
    V4_Y_NORM_IDX,
)
from models.focus_priors import _parse_money as _PRIORS_PARSE_MONEY
from models.postprocess_company import _company_span
from models.rule_fields import (
    _SUBTOTAL_KW_RE,
    _TOTAL_KW_RE,
    extract_date,
    extract_total,
)
from models.rule_regex import DATE_RE, MONEY_RE, repair_money_ocr
from models.total_arithmetic import total_arithmetic_consensus
from models.total_post import apply_zone_gate, extract_total_value
from models.zone_prior import decode_zone_posterior

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    import torch

# FOCUS-C span head — anchor and boundary regex patterns.
_COMPANY_ANCHOR = re.compile(
    r"\b(SDN\.?\s*BHD|BERHAD|ENTERPRISE|RESTAURANT|TRADING|CORPORATION|"
    r"HOLDINGS|MARKETING|GROUP|CAFE|RESOURCES|MART|MAJU|JAYA|SERVICES|"
    r"S/B|\(M\)\s*SDN)\b",
    re.IGNORECASE,
)
_COMPANY_REG_ID_RE = re.compile(
    r"\b(NO\.?\s*)?[0-9]{5,}-?[A-Z]?\b|\bROC\s*NO\b|\bGST\s*ID\b|"
    r"\bCO\.?\s*NO\b|\bID\s*NO\b",
    re.IGNORECASE,
)
_COMPANY_HEADER_RE = re.compile(
    r"^(TAX\s+INVOICE|CASH\s+RECEIPT|OFFICIAL\s+RECEIPT|RECEIPT|INVOICE|"
    r"WELCOME|TERIMA\s+KASIH|THANK\s+YOU)\b",
    re.IGNORECASE,
)
_COMPANY_UPPER_CONT_RE = re.compile(r"^[A-Z][A-Z0-9 \-&.'()/]+$")
# Wider continuation regex used by the back-extension only: also
# accepts lines that *start* with ``&``, ``(``, or a digit so trade
# names like ``"& RUNCIT"`` (continuation of a multi-line registered
# name) and ``"99 SPEED MART"`` are not stranded above the SDN BHD
# anchor.  Kept separate from :data:`_COMPANY_UPPER_CONT_RE` so the
# forward-extension semantics — which require an upper-case anchor —
# do not change.
_COMPANY_BACK_CONT_RE = re.compile(r"^[A-Z0-9&(][A-Z0-9 \-&.'()/]*$")


def _is_company_boundary(text: str) -> bool:
    """Return True for lines that should NOT anchor/extend a company span.

    Unlike address boundaries, company names (SDN BHD, ENTERPRISE, etc.) are
    NOT boundaries here — they ARE company spans. Boundaries are:
    - Registration IDs like (123456-W)
    - Receipt headers like RECEIPT, INVOICE, etc.
    - Money values
    - Date values
    - Phone/fax numbers
    """
    from models.rule_regex import _DATE_RE, _MONEY_RE
    s = text.strip()
    if not s:
        return True
    if _COMPANY_REG_ID_RE.search(s):
        return True
    if _COMPANY_HEADER_RE.match(s):
        return True
    if _MONEY_RE.search(s) or _DATE_RE.search(s):
        return True
    # Phone/fax lines are boundaries
    phone_re = re.compile(r"(?:TEL|FAX|PHONE)[:\s]*\d", re.IGNORECASE)
    if phone_re.search(s):
        return True
    return False


def _company_anchor_filter(
    picks: list[int], texts: list[str], bboxes: list[list[float]],
) -> list[int]:
    """Filter FOCUS-C span picks via lexical anchor + contiguity rules.

    1. Sort picks top-down by ``bboxes[i][1]`` (y1).
    2. Drop boundary lines (``_is_company_boundary``) from picks.
    3. Find topmost remaining pick whose text matches ``_COMPANY_ANCHOR``;
       if none, return ``picks[:1]`` (no-anchor fallback: keep first only).
    4. Extend backward 1 line: if the pick immediately before the anchor
       in the original picks list is a non-boundary short alpha line
       (≤6 tokens, all letters/spaces), include it as parent brand.
    5. Extend forward only while the next pick (after anchor in original
       picks) hits ``_COMPANY_ANCHOR`` OR matches ``_COMPANY_UPPER_CONT_RE``.
       Stop on first non-matching line.
    6. Return the resulting contiguous sub-list in original picks order.
    """
    if not picks or not texts or not bboxes:
        return []
    # Sort picks top-down.
    sorted_picks = sorted(picks, key=lambda i: bboxes[i][1] if i < len(bboxes) else 999)
    # Drop boundary lines.
    clean = [i for i in sorted_picks if not _is_company_boundary(texts[i])]
    if not clean:
        return picks[:1] if picks else []
    # Find anchor.
    anchor_idx_in_clean: int | None = None
    for ci, pi in enumerate(clean):
        if _COMPANY_ANCHOR.search(texts[pi]):
            anchor_idx_in_clean = ci
            break
    if anchor_idx_in_clean is None:
        return clean[:1]
    anchor_pick = clean[anchor_idx_in_clean]
    result = [anchor_pick]
    # Extend backward (parent brand) — multi-line walk with a relaxed
    # character set covering trade names that contain ``&``, ``.``,
    # digits, hyphens, or parentheses (e.g.\ ``"99 SPEED MART"``,
    # ``"K-DESIGN"``, ``"KEDAI UBAT & RUNCIT HONG NING"``).  The legacy
    # 1-line, alpha+space-only filter chronically under-extended the
    # span on Malaysian receipts whose registered name spans 3–5 lines
    # above the SDN BHD anchor (the dominant ``wrong_span`` mode in
    # the Fig.~8 error decomposition).  Cap at 4 lines back so a noisy
    # top-of-receipt header band cannot pollute the company span.
    for ci in range(anchor_idx_in_clean - 1, -1, -1):
        if len(result) - 1 >= 4:
            break
        prev_pick = clean[ci]
        prev_text = texts[prev_pick].strip()
        if not prev_text:
            break
        if (
            _COMPANY_HEADER_RE.search(prev_text)
            or _COMPANY_REG_ID_RE.search(prev_text)
        ):
            break
        if not _COMPANY_BACK_CONT_RE.match(prev_text):
            break
        result.insert(0, prev_pick)
    # Extend forward.
    for ci in range(anchor_idx_in_clean + 1, len(clean)):
        cand = clean[ci]
        t = texts[cand].strip()
        if _COMPANY_ANCHOR.search(t) or _COMPANY_UPPER_CONT_RE.match(t):
            result.append(cand)
        else:
            break
    # Return in original picks order.
    pick_set = set(result)
    return [p for p in picks if p in pick_set]


@dataclass(frozen=True)
class AssignerPolicy:
    """Inference-time policy knobs for the learned assigner (PR-A / T-D1).

    Promoted from a 7-kwarg surface on :func:`_assign_learned_with_attn`
    so the inference path matches the 2-in/1-out contract documented in
    ``AGENTS.md``.  All knobs are typed and immutable; callers pass the
    same :class:`AssignerPolicy` to every receipt for consistency.

    Field defaults mirror the legacy hard-coded values so an unchanged
    eval run reproduces bit-for-bit.
    """

    regex_router: bool = True
    address_accept_fraction: float = 0.5
    total_confidence_threshold: float = 0.55
    multi_line_fraction: float = 0.5
    total_override_margin: float = 0.10
    total_override_margin_diffuse: float = 0.05
    attn_blend_alpha: float = 0.5
    attn_log_eps: float = 1.0e-6
    # PR-C / S0 — address-assembly scoring weights (used by the
    # ``models.pipeline_consensus_score._score_address_assembly`` head
    # when ``address_score_token_f1_w > 0``).
    address_score_token_f1_w: float = 1.0
    address_score_line_count_w: float = 0.25
    address_score_postcode_w: float = 0.05
    address_score_money_penalty: float = 0.10
    # PR #113 / H1 — FOCUS-A address-span dispatch gate.  Default 0.0
    # keeps callers that omit the knob bit-exact (legacy threshold-band
    # path), but ``configs/default.json`` sets ``focus_confidence_floor``
    # on :class:`ExpConfig` to 0.10 so eval runs gate the trained head
    # at the deployment threshold.  Used by :func:`_assign_learned_with_attn`
    # to decide when to fall back to ``_legacy_address_pick``.
    focus_confidence_floor: float = 0.0
    # FOCUS-C company-anchor dispatch gate.  ``0.0`` keeps the legacy
    # cross-attn argmax for the ``company`` field; ``configs/default.json``
    # routes ``focus_company_confidence_threshold`` (0.30) onto the
    # :class:`ExpConfig` so eval gates the trained head at the
    # deployment threshold (mirrors :attr:`focus_confidence_floor`).
    focus_company_confidence_threshold: float = 0.0
    # FOCUS-C span dispatch gate.  When the span head confidence is at
    # or above this floor, the span path fires; below falls through to
    # the single-line ``company_pick`` argmax path.
    focus_company_confidence_floor: float = 0.0
    # FOCUS-T arithmetic consensus gate.  When True (default), the
    # ``total`` dispatch tries :func:`models.total_arithmetic.
    # total_arithmetic_consensus` *before* the regex router so receipts
    # whose grand total satisfies ``cash − change`` or
    # ``subtotal + tax + service − discount`` commit the consensus
    # value without trusting the (often OCR-corrupted) total line.
    # ``False`` reproduces the legacy regex-router-first dispatch
    # bit-for-bit.
    total_arithmetic_enabled: bool = True
    # Relational receipt-zone prior — when enabled, a 3-state monotone
    # HMM (header → items → totals) is decoded by forward–backward over
    # per-line text features and routed into both the FOCUS-C company
    # dispatch (additive log-``p_header`` on the company attention row,
    # abstention gate on ``company_pick``) and the FOCUS-T total
    # dispatch (filter candidate-money lines by ``p_total >=
    # totals_zone_floor``; drop ``p_total < regex_total_floor`` from
    # the regex-argmax fallback).  Defaults preserve legacy bit-for-bit
    # behaviour when ``zone_cfg.enabled`` is False.
    zone_cfg: ZoneConfig = field(default_factory=lambda: ZoneConfig(enabled=False))


@dataclass(frozen=True)
class AssignerInputs:
    """Per-receipt inputs to :func:`_assign_learned_with_attn` (PR-A / T-D1).

    Holds the four lists/tensors that previously flowed through the
    7-kwarg signature.  ``device`` is a ``torch.device`` so callers do
    not have to remember the legacy ``str`` form.
    """

    texts: list[str]
    feats: list[torch.Tensor]
    bboxes: list[list[float]]
    fields: list[str]
    device: str


DEFAULT_POLICY = AssignerPolicy()

# Fields whose GT value spans multiple OCR regions (address = street/city/
# postcode).  Pick every region with ``attn >= _MULTI_LINE_FRACTION * max``
# and concatenate top→bottom; pos-mass loss trains for this.  The
# fraction was ``0.5`` originally, which drops the 3rd/4th line of a
# long address whenever attention is spread even slightly — 38/63
# address misses in the live miss table are pure prefix-of-GT.  ``0.25``
# widens the accept band; ``refine_assignments`` then filters junk
# post-hoc via ``_is_addr_boundary`` so the lower threshold never
# pulls in phone/tax-id/header lines.
_MULTI_LINE_FIELDS = frozenset({"address"})
# Change B — default restored from 0.25 to 0.5.  The 0.25 band pulled
# phone/tax/GST lines into the address whenever the attention head was
# mildly diffuse (the common failure mode of a 2-layer encoder trained
# on 500 SROIE receipts); the new ``enforce_address_contiguity`` gate
# below now excises such lines spatially, so we no longer need the
# wider band to compensate for missed prefix-of-GT lines.  Callers that
# want a different band pass ``address_accept_fraction`` via
# :class:`ExpConfig` (Change G).
_MULTI_LINE_FRACTION = 0.5

_FIELD_REGEX = {"date": DATE_RE, "total": MONEY_RE}


def _build_priors(
    texts: list[str], bboxes: list[list[float]], n_priors: int,
) -> list[list[float]]:
    """Per-region text priors matching the assigner's expected dim (6/9/14/20).

    v2 mirrors :mod:`models.assigner_data`: ``y_norm = bbox[3] / max_y`` and
    ``is_last_money_line = (i == argmax_i(_MONEY_RE.search(texts[i])))``.
    v3 extends v2 with five distractor-aware bits (SUBTOTAL / CASH /
    CHANGE / TAX / ROUNDING) — strategy E of the assigner plan.
    v4 extends v3 with the FOCUS-T/C structural priors (FOCUS framework,
    paper §III-D rewrite): is_subtotal_kw, is_tax_kw,
    is_company_boilerplate, line_y_normalised, money_value_normalised,
    arithmetic_witness_self.  Unknown ``n_priors`` raise ``ValueError``
    (no silent zero-padding).
    """
    if n_priors == N_TEXT_PRIORS:
        return [text_priors(t) for t in texts]
    if n_priors in (N_TEXT_PRIORS_V2, N_TEXT_PRIORS_V3, N_TEXT_PRIORS_V4):
        money_idxs = [i for i, t in enumerate(texts) if _PRIORS_MONEY_RE.search(t)]
        last_money = max(money_idxs) if money_idxs else -1
        y_vals = [bb[3] for bb in bboxes]
        denom = max(max(y_vals) if y_vals else 1.0, 1e-6)
        if n_priors == N_TEXT_PRIORS_V2:
            return [
                text_priors_v2(texts[i], bboxes[i][3] / denom, i == last_money)
                for i in range(len(texts))
            ]
        if n_priors == N_TEXT_PRIORS_V3:
            return [
                text_priors_v3(texts[i], bboxes[i][3] / denom, i == last_money)
                for i in range(len(texts))
            ]
        # v4 — FOCUS-T/C: receipt-level witness column + money_value_norm.
        witnesses = arithmetic_witnesses_v4(texts)
        monies = [_PRIORS_PARSE_MONEY(t) for t in texts]
        max_money = max((m for m in monies if m is not None), default=0.0)
        money_denom = max(max_money, 1e-6)
        money_norm = [
            (m / money_denom) if m is not None else 0.0 for m in monies
        ]
        return [
            text_priors_v4(
                texts[i], bboxes[i][3] / denom, i == last_money,
                money_norm[i], witnesses[i],
            )
            for i in range(len(texts))
        ]
    raise ValueError(
        f"Unsupported n_text_priors={n_priors}; "
        f"expected {N_TEXT_PRIORS}, {N_TEXT_PRIORS_V2}, "
        f"{N_TEXT_PRIORS_V3}, or {N_TEXT_PRIORS_V4}.",
    )


def postprocess_value(name: str, value: str) -> str:
    """Strip region text to SROIE GT format; for ``total`` retry after OCR-repair.

    Fix 5 — ``total`` routes through :func:`extract_total_value` which
    strictly anchors to the rightmost ``\\d{1,3}(,\\d{3})*\\.\\d{2}`` on
    the line and only falls back to a lenient ``\\d+\\.\\d{2}`` match
    when TrOCR has dropped the thousands separator (``"RM I15.00"``).
    Other fields use the legacy ``_FIELD_REGEX`` strip.
    """
    if name == "total":
        return extract_total_value(value)
    pattern = _FIELD_REGEX.get(name)
    if pattern is None:
        return value
    m = pattern.search(value)
    if not m:
        return value
    return m.group(0).strip()


def _apply_date_sanity(value: str, texts: list[str]) -> str:
    """Fix C — reject implausible SROIE dates, fall back to OCR scan.

    ``value`` is the post-:func:`postprocess_value` ``date`` string
    (already regex-stripped to a date-shaped substring by the caller).
    If :func:`models.date_postprocess.is_plausible` rejects it (year
    outside 2014–2019, malformed day/month), scan ``texts`` for the
    first plausible ``DD/MM/YYYY`` and return that instead.  When no
    plausible alternative exists, return the original value unchanged
    so the downstream normaliser still has something to canonicalise
    (strictly no worse than the pre-Fix behaviour).
    """
    if is_plausible(value):
        return value
    alt = fallback_from_ocr_lines(texts)
    return alt if alt is not None else value


def _has_regex_value(name: str, text: str) -> bool:
    """True iff ``text`` contains a valid regex value for field ``name``."""
    pattern = _FIELD_REGEX.get(name)
    if pattern is None:
        return True
    return bool(pattern.search(text) or (
        name == "total" and pattern.search(repair_money_ocr(text))))


def _route_regex_field(
    name: str, texts: list[str], bboxes: list[list[float]], used: set[int],
) -> tuple[int, str] | None:
    """Change A — regex-oracle router for deterministic fields (date/total).

    Invert the legacy "attention picks region, regex filters the string"
    pipeline: ask the rule-based extractor first.  Only return a pick
    when it is *unambiguous* — otherwise fall through so the attention
    argmax handles the genuinely hard case.

    * ``date``  — first DATE_RE match in reading order (already unique
                  on well-formed SROIE receipts).
    * ``total`` — confident iff the rule extractor's pick has a TOTAL
                  keyword in its ±1-line neighbourhood; ambiguous
                  (multiple money lines with *no* TOTAL keyword anywhere)
                  falls through to attention.

    Returns ``None`` on ambiguity or when the pick was already consumed
    by another field.
    """
    if name == "date":
        pick = extract_date(texts)
    elif name == "total":
        pick = extract_total(texts, bboxes)
    else:
        return None
    if pick is None or pick[0] in used:
        return None
    if name == "total" and not _is_confident_total(texts, pick[0]):
        return None
    return pick


def _is_confident_total(texts: list[str], idx: int) -> bool:
    """A ``total`` pick is confident iff its ±1-line window contains a
    TOTAL keyword and the picked line itself is not a SUBTOTAL line.

    Returns ``False`` for out-of-range ``idx`` as a defensive guard.
    Co-occurrence of SUBTOTAL in a *neighbour* line (the canonical
    SROIE layout is ``SUBTOTAL … / TOTAL …`` on consecutive rows) is
    NOT a disqualifier; we only disqualify when the picked line itself
    is labelled SUBTOTAL or when no TOTAL keyword appears at all in
    the neighbourhood (the "multiple money lines, no TOTAL keyword"
    ambiguity case from the Change A spec).
    """
    if idx >= len(texts):
        return False
    if _SUBTOTAL_KW_RE.search(texts[idx]):
        return False
    lo, hi = max(0, idx - 1), min(len(texts), idx + 2)
    window = " ".join(texts[lo:hi])
    return bool(_TOTAL_KW_RE.search(window))


def _confidence_gate_total(
    w: torch.Tensor, best: int, texts: list[str],
    bboxes: list[list[float]], used: set[int], threshold: float,
) -> tuple[int, str] | None:
    """Fix 3 — return a rule-based ``total`` pick when the assigner is
    unconfident.

    Triggers when either:
      * ``softmax(w).max() < threshold`` — the head is spread across
        multiple money lines with no clear winner (the 34-miss
        subtotal/rounding/change confusion mode), or
      * the attention's argmax line itself matches the SUBTOTAL /
        SERVICE / TENDER keyword regex — the assigner chose a known
        distractor.

    Returns ``None`` when the gate should NOT fire (assigner is
    confident and its pick is not a distractor), leaving the caller's
    attention-argmax path intact.
    """
    if best < 0 or best >= len(texts):
        return None
    picked_is_distractor = bool(_SUBTOTAL_KW_RE.search(texts[best]))
    softmax_max = float(torch.softmax(w, dim=-1).max().item())
    if softmax_max >= threshold and not picked_is_distractor:
        return None
    pick = extract_total(texts, bboxes)
    if pick is None or pick[0] in used:
        return None
    return pick


def _legacy_address_pick(
    attn_row: torch.Tensor, texts: list[str],
    bboxes: list[list[float]], used: set[int], frac: float,
) -> tuple[list[int], str]:
    """Legacy threshold-band → :func:`enforce_address_contiguity` → join.

    Extracted as a private helper (PR #113 / H1) so the FOCUS-A
    :meth:`AttentionAssigner.address_span` dispatch in
    :func:`_assign_learned_with_attn` can fall back to this exact chain
    when the span head is absent or abstains.  Returns the picked
    indices (top→bottom by ``bboxes[i][1]``) and the joined text.

    PR-ADDR-PREC adds a boundary filter to the threshold band: any
    region whose text matches :func:`models.consensus._is_addr_boundary`
    (money / date / phone / GST / company-header / invoice-cashier
    transition) or :data:`models.rule_regex._ADDR_LEADING_JUNK_RE`
    (CO. NO., GST NO., REG NO., …) is dropped *before* the spatial
    contiguity gate.  This stops the wider 0.5 accept-band from
    pulling ``BHD`` / ``INTERNATIONAL`` / ``INV NO`` / ``CASH`` /
    ``RECEIPT`` / ``DOC NO`` lines into the address span on
    diffuse-attention receipts (the precision-drop pattern dominating
    the live miss table).
    """
    from models.rule_regex import _ADDR_LEADING_JUNK_RE
    max_w = float(attn_row.max().item())
    if max_w <= 0:
        return [], ""
    stripped = [t.strip() for t in texts]
    picks = [
        i for i in range(attn_row.shape[0])
        if i not in used and float(attn_row[i].item()) >= frac * max_w
    ]
    # Boundary filter — drop picks classified as boundary lines or
    # leading-junk tax-ID / reg-no headers before contiguity.
    picks = [
        i for i in picks
        if not _consensus_is_addr_boundary(stripped[i])
        and not _ADDR_LEADING_JUNK_RE.match(stripped[i])
    ]
    if not picks:
        return [], ""
    picks.sort(key=lambda i: bboxes[i][1])
    picks = enforce_address_contiguity(picks, bboxes)
    value = " ".join(stripped[i] for i in picks if stripped[i])
    return picks, value


def _witness_gate_total(
    prior_list: list[list[float]], texts: list[str],
    used: set[int], n_priors: int,
) -> tuple[int, str] | None:
    """Hard arithmetic witness gate for the total field.

    When exactly one unambiguous line satisfies subtotal + tax == total
    (within 2¢, encoded as ``priors_v4[V4_WITNESS_IDX] == 1.0``), return
    it directly without consulting attention or the keyword-based rule
    extractor.  Falls through on 0 or >1 witness hits so ambiguous
    receipts are not mis-assigned.

    Requires priors_v4 (n_priors >= V4_WITNESS_IDX + 1); no-ops silently
    on v1/v2/v3 priors so old checkpoints stay bit-exact.
    """
    if n_priors <= V4_WITNESS_IDX:
        return None
    hits = [
        i for i in range(len(prior_list))
        if i not in used and prior_list[i][V4_WITNESS_IDX] >= 1.0
    ]
    if len(hits) != 1:
        return None
    idx = hits[0]
    return idx, texts[idx]


def _assign_learned(
    assigner: AttentionAssigner, texts: list[str],
    feats: list[torch.Tensor], bboxes: list[list[float]],
    fields: list[str], device: str,
) -> dict[str, str]:
    """Use AttentionAssigner cross-attention to pick regions per field."""
    values, _attn = _assign_learned_with_attn(
        assigner, texts, feats, bboxes, fields, device,
    )
    return values


def _assign_learned_with_attn(
    assigner: AttentionAssigner, texts: list[str],
    feats: list[torch.Tensor], bboxes: list[list[float]],
    fields: list[str], device: str,
    address_accept_fraction: float | None = None,
    regex_router: bool = True,
    total_confidence_threshold: float = 0.0,
    focus_confidence_floor: float = 0.0,
    focus_company_confidence_threshold: float = 0.0,
    focus_company_confidence_floor: float = 0.0,
    total_arithmetic_enabled: bool = False,
    zone_cfg: ZoneConfig | None = None,
) -> tuple[dict[str, str], torch.Tensor | None]:
    """Field-assign and return (F, N) cross-attention for fig_attn_heatmap.

    Multi-line fields (address) are dispatched in two layers (PR #113
    / H1):

    * If the assigner carries a trained FOCUS-A span head
      (``assigner._span_head is not None``), call
      :meth:`AttentionAssigner.address_span` over the per-receipt
      post-encoder ``kv`` slice and accept its prediction when
      ``j >= i`` and ``confidence >= focus_confidence_floor``.  This
      crops the over-prediction band that drives address P down on
      diffuse-head receipts.
    * Otherwise (or when the span head abstains), fall back to the
      legacy threshold-band → :func:`enforce_address_contiguity`
      chain via :func:`_legacy_address_pick` so ``focus_enabled=False``
      runs reproduce bit-for-bit.

    The ``company`` field gets an analogous FOCUS-C dispatch when the
    assigner carries a trained company head
    (``assigner.focus_company_enabled``):
    :meth:`AttentionAssigner.company_pick` returns the merchant trade-
    name anchor index and its softmax confidence; when ``confidence
    >= focus_company_confidence_threshold`` the
    :func:`models.postprocess_company._company_span` greedy assembler
    forward-extends 0..2 ``SDN BHD`` / ``(REG-NO)`` suffix lines.
    Below the threshold (or when the head is absent) the legacy
    cross-attn argmax pick wins so confident negatives never regress.

    Additionally, when the FOCUS-C span head is enabled
    (``focus_company_span_enabled``), the span head prediction is tried
    first and filtered via :func:`_company_anchor_filter`; the existing
    ``company_pick`` + ``_company_span`` path is the fallback.

    Regex-deterministic fields (date/total) are resolved via
    :func:`_route_regex_field` first when ``regex_router`` is True
    (Change A); attention is the fallback.  When
    ``total_confidence_threshold > 0`` the attention fallback for
    ``total`` additionally goes through :func:`_confidence_gate_total`
    (Fix 3): a low-confidence or SUBTOTAL-flagged pick is replaced by
    the rule-based extractor.  Returns ``(values, None)`` on empty text.
    """
    if not texts:
        return {}, None
    tf = torch.cat(feats, dim=0).unsqueeze(0).to(device)
    bf = torch.tensor(bboxes, dtype=torch.float32).unsqueeze(0).to(device)
    prior_list = _build_priors(texts, bboxes, assigner.n_text_priors)
    priors = torch.tensor(
        prior_list, dtype=torch.float32,
    ).unsqueeze(0).to(device)
    # Zone-prior — relational 3-state (header / items / totals) HMM
    # decoded once per receipt and routed into both the FOCUS-C company
    # dispatch and the FOCUS-T total dispatch below.  Empty / disabled
    # config returns a uniform posterior so legacy callers reproduce
    # bit-for-bit.  Requires v4 priors for the y_norm + boilerplate
    # columns; older priors fall back to text-only features (y=0).
    n_pri = len(prior_list[0]) if prior_list else 0
    zcfg = zone_cfg if zone_cfg is not None else ZoneConfig(enabled=False)
    if zcfg.enabled and n_pri > V4_IS_COMPANY_BOILERPLATE_IDX:
        zone_lines = [
            (
                texts[i],
                float(prior_list[i][V4_Y_NORM_IDX]),
                float(prior_list[i][V4_IS_COMPANY_BOILERPLATE_IDX]),
            )
            for i in range(len(texts))
        ]
        zone_post = decode_zone_posterior(zone_lines, zcfg)
    else:
        zone_post = []
    p_header: list[float] = [p[0] for p in zone_post]
    p_totals: list[float] = [p[2] for p in zone_post]
    # PR #113 / H1 + FOCUS-C — request ``kv`` whenever a head needs
    # the post-encoder tensor (FOCUS-A span head OR FOCUS-C positional
    # head OR FOCUS-C span head); when all heads are disabled
    # (focus_company_span_enabled=False by default), the no-head path
    # stays bit-identical to the legacy ``assigner(...)`` call.
    kv: torch.Tensor | None
    needs_kv = (
        assigner._span_head is not None
        or getattr(assigner, "focus_company_enabled", False)
        or getattr(assigner, "focus_company_span_enabled", False)
    )
    if needs_kv:
        _logits, attn_w, kv = assigner.forward_with_kv(tf, bf, priors)
    else:
        _logits, attn_w = assigner(tf, bf, priors)
        kv = None
    attn_sample = attn_w[0].detach().cpu()  # (F, N), kept for the sampler
    used: set[int] = set()
    out: dict[str, str] = {}
    addr_frac = (
        address_accept_fraction if address_accept_fraction is not None
        else _MULTI_LINE_FRACTION
    )
    for f_idx, name in enumerate(fields):
        if len(used) >= len(texts):
            break
        # FOCUS-T arithmetic consensus — runs ahead of every other
        # ``total`` path.  When ≥2 of {cash−change, sub+tax+svc−disc}
        # agree to ±2¢, or exactly one fires unambiguously, the
        # consensus value is committed without consulting the regex
        # router or attention.  ``idx == -1`` signals a synthesised
        # value (no OCR line carries it verbatim — recovers OCR-
        # corrupted total lines), in which case no line is consumed
        # in ``used`` so the residual money lines remain available
        # to other fields' fallbacks.  Falls through silently when
        # under-determined; legacy callers (``total_arithmetic_enabled
        # =False``) bypass this block entirely.
        if total_arithmetic_enabled and name == "total":
            ar = total_arithmetic_consensus(
                texts, used,
                p_totals=p_totals if p_totals else None,
                totals_zone_floor=zcfg.totals_zone_floor,
            )
            if ar is not None:
                ar_idx, ar_value = ar
                if ar_idx >= 0:
                    used.add(ar_idx)
                out[name] = postprocess_value(name, ar_value)
                continue
        # Change A — regex-oracle router for date/total.
        if regex_router and name in _FIELD_REGEX:
            routed = _route_regex_field(name, texts, bboxes, used)
            if routed is not None:
                best_idx, value = routed
                used.add(best_idx)
                pv = postprocess_value(name, value)
                if name == "date":
                    pv = _apply_date_sanity(pv, texts)
                out[name] = pv
                continue
        w = attn_w[0, f_idx].clone()
        for u in used:
            w[u] = -1e9
        # Zone-prior bias for the ``company`` row of cross-attention:
        # add ``log p_header`` per line so the argmax fallback (and
        # any consumer of ``w``) is biased toward the header zone.
        # Bit-exact when ``p_header`` is uniform (zone prior disabled
        # or abstaining), strict improvement when the prior
        # concentrates mass on the actual header lines.
        if name == "company" and p_header:
            for li in range(min(len(p_header), int(w.shape[0]))):
                w[li] = w[li] + math.log(max(p_header[li], 1e-6))
        # FOCUS-T regex-argmax fallback (the deterministic ``total``
        # path below) consults ``used`` to skip already-consumed lines;
        # extending ``used`` with lines whose ``p_total`` is below the
        # configured regex floor reroutes the same dropout semantics
        # for header-zone numerics (phone / regid digits the money
        # regex would otherwise accept).
        if name == "total" and p_totals:
            used = apply_zone_gate(used, p_totals, zcfg.regex_total_floor)
            # Re-blank the augmented ``used`` set on ``w`` so the
            # downstream ``argsort(w, descending=True)`` skips them.
            for u in used:
                if u < int(w.shape[0]):
                    w[u] = -1e9
        if name in _MULTI_LINE_FIELDS:
            # PR #113 / H1 — FOCUS-A span dispatch.  The trained head
            # crops the diffuse threshold band to a contiguous
            # ``[i, j]`` interval; we accept it iff it returned a
            # well-formed span at or above the configured confidence
            # floor.  Otherwise fall through to the legacy chain.
            span_value: str | None = None
            picks: list[int] = []
            if assigner._span_head is not None and kv is not None:
                pred = assigner.address_span(kv[0], texts)
                if (
                    pred["j"] >= pred["i"]
                    and pred["confidence"] >= focus_confidence_floor
                ):
                    # PR-ADDR-PREC-2 — deterministic shrink-from-both-ends.
                    # The trained span head has occasional residual leak
                    # at the endpoints (a borderline header line that the
                    # boundary penalty did not fully repel).  This loop
                    # walks ``i`` forward and ``j`` backward while the
                    # endpoint matches the broadened address-boundary
                    # classifier; ``(0, -1)`` collapses through the
                    # legacy fallback.
                    si, sj = shrink_addr_span(
                        (pred["i"], pred["j"]), texts,
                    )
                    if sj >= si:
                        picks = list(range(si, sj + 1))
                        span_value = " ".join(
                            t for t in (texts[k].strip() for k in picks) if t
                        )
            if span_value is None:
                picks, span_value = _legacy_address_pick(
                    w, texts, bboxes, used, addr_frac,
                )
            if not picks:
                continue
            for i in picks:
                used.add(i)
            value = span_value
        else:
            # FOCUS-C span dispatch: when the span head is configured,
            # try it first; filter via ``_company_anchor_filter``.
            # Fall through to the single-line ``company_pick`` path
            # when the span head is absent or unconfident.
            if (
                name == "company"
                and getattr(assigner, "focus_company_span_enabled", False)
                and assigner._company_span_head is not None
                and kv is not None
            ):
                cspan = assigner.company_span(kv[0], texts)
                if (
                    cspan["j"] >= cspan["i"]
                    and cspan["confidence"] >= focus_company_confidence_floor
                ):
                    cpicks = list(range(cspan["i"], cspan["j"] + 1))
                    cpicks = _company_anchor_filter(cpicks, texts, bboxes)
                    cpicks = [i for i in cpicks if i not in used]
                    cspan_value = " ".join(
                        t for t in (texts[i].strip() for i in cpicks) if t
                    )
                    if cpicks and cspan_value:
                        for i in cpicks:
                            used.add(i)
                        out[name] = postprocess_value(name, cspan_value)
                        continue
            # FOCUS-C dispatch: replace the cross-attn argmax for the
            # ``company`` field with :meth:`AttentionAssigner.company_pick`
            # over the post-encoder ``kv``, then forward-extend through
            # :func:`models.postprocess_company._company_span`.  The
            # ``focus_company_confidence_threshold`` gate falls back to
            # the legacy argmax when the head is unconfident so confident
            # negatives are never regressed.  Mirrors the FOCUS-A
            # address-span dispatch above.
            if (
                name == "company"
                and getattr(assigner, "focus_company_enabled", False)
                and assigner._company_head is not None
                and kv is not None
                and assigner.n_text_priors == N_TEXT_PRIORS_V4
            ):
                y_col = priors[0, :, V4_Y_NORM_IDX]
                bp_col = priors[0, :, V4_IS_COMPANY_BOILERPLATE_IDX]
                cpred = assigner.company_pick(kv[0], texts, y_col, bp_col)
                # Zone-prior abstention: if the predicted line is not
                # in the header zone (``p_header`` below the configured
                # floor), treat as abstention so the legacy fallback
                # below can fire.  Bit-exact when the zone prior is
                # disabled (``p_header`` empty) or abstains (uniform).
                cidx = cpred["i"]
                in_header_zone = (
                    not p_header
                    or cidx < 0
                    or cidx >= len(p_header)
                    or p_header[cidx] >= zcfg.header_zone_floor
                )
                if (
                    cpred["i"] >= 0
                    and cpred["confidence"] >= focus_company_confidence_threshold
                    and in_header_zone
                ):
                    bp_flags = [bool(prior_list[i][V4_IS_COMPANY_BOILERPLATE_IDX])
                                for i in range(len(texts))]
                    picks, span_value = _company_span(
                        texts, bboxes, bp_flags, anchor_idx=cpred["i"],
                    )
                    if picks and span_value:
                        for i in picks:
                            used.add(i)
                        out[name] = postprocess_value(name, span_value)
                        continue
            # FOCUS-T hard arithmetic witness gate: when exactly one line
            # satisfies subtotal + tax == total (within 2¢) and is not yet
            # used, pick it unconditionally.  Fires before attention argmax
            # so the arithmetic relationship overrides a diffuse head.
            # Falls through to attention when the witness is ambiguous (0
            # or >1 hits) so behaviour on non-standard receipts is unchanged.
            if name == "total":
                witnessed = _witness_gate_total(
                    prior_list, texts, used, assigner.n_text_priors,
                )
                if witnessed is not None:
                    best, value = witnessed
                    used.add(best)
                    out[name] = postprocess_value(name, value)
                    continue
            # Regex fields (total/date): argmax with runner-up fallback so
            # a label-only pick (``"TOTAL:"``) doesn't score F1=0.
            if name in _FIELD_REGEX:
                order = [int(i) for i in torch.argsort(w, descending=True).tolist()]
                best = next(
                    (i for i in order
                     if i not in used and _has_regex_value(name, texts[i])),
                    order[0],
                )
                # Fix 3 — confidence gate for ``total``: low softmax max
                # or a SUBTOTAL-keyword pick delegates to the rule-based
                # extractor.  ``total_confidence_threshold <= 0`` disables
                # the gate, keeping bit-compat with legacy callers.
                if name == "total" and total_confidence_threshold > 0:
                    gated = _confidence_gate_total(
                        w, best, texts, bboxes, used, total_confidence_threshold,
                    )
                    if gated is not None:
                        best, value = gated
                        used.add(best)
                        out[name] = postprocess_value(name, value)
                        continue
            else:
                best = int(w.argmax().item())
                # Heuristic span fallback for ``company`` when no FOCUS head
                # fired (checkpoint lacks trained company/span heads).  Feed
                # the argmax as the anchor into the greedy ``_company_span``
                # assembler so multi-line company names (e.g. "UNIHAKKA
                # INTERNATIONAL" + "SDN BHD" on separate lines) are joined
                # rather than truncated to the single highest-attention line.
                # Only activates when ``company`` and both FOCUS-C heads are
                # absent; bit-exact when either head fires (``continue`` above).
                if name == "company" and best not in used:
                    n_pri = len(prior_list[0]) if prior_list else 0
                    bp_flags = [
                        bool(prior_list[i][V4_IS_COMPANY_BOILERPLATE_IDX])
                        if n_pri > V4_IS_COMPANY_BOILERPLATE_IDX else False
                        for i in range(len(texts))
                    ]
                    span_picks, span_val = _company_span(
                        texts, bboxes, bp_flags, anchor_idx=best,
                    )
                    span_picks = [i for i in span_picks if i not in used]
                    if span_picks and span_val:
                        for i in span_picks:
                            used.add(i)
                        out[name] = postprocess_value(name, span_val)
                        continue
            used.add(best)
            value = texts[best]
        pv = postprocess_value(name, value)
        if name == "date":
            pv = _apply_date_sanity(pv, texts)
        out[name] = pv
    return out, attn_sample


def assign_with_policy(
    assigner: AttentionAssigner,
    inputs: AssignerInputs,
    policy: AssignerPolicy = DEFAULT_POLICY,
) -> tuple[dict[str, str], torch.Tensor | None]:
    """PR-A / T-D1 — 2-in/1-out wrapper for :func:`_assign_learned_with_attn`.

    The legacy 7-kwarg signature is preserved for one PR (callers that
    haven't yet migrated to the dataclass form continue to work);
    PR-C will delete the 7-kwarg form and route every caller through
    this entry point.

    Returns ``(values, attn_sample)`` exactly like the legacy form.
    """
    return _assign_learned_with_attn(
        assigner, inputs.texts, inputs.feats, inputs.bboxes,
        inputs.fields, inputs.device,
        address_accept_fraction=policy.address_accept_fraction,
        regex_router=policy.regex_router,
        total_confidence_threshold=policy.total_confidence_threshold,
        focus_confidence_floor=policy.focus_confidence_floor,
        focus_company_confidence_threshold=(
            policy.focus_company_confidence_threshold
        ),
        focus_company_confidence_floor=(
            policy.focus_company_confidence_floor
        ),
        total_arithmetic_enabled=policy.total_arithmetic_enabled,
        zone_cfg=policy.zone_cfg,
    )
