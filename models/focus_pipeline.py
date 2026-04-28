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

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
from models.focus_priors import _parse_money as _PRIORS_PARSE_MONEY
from models.rule_fields import (
    _SUBTOTAL_KW_RE,
    _TOTAL_KW_RE,
    extract_date,
    extract_total,
)
from models.rule_regex import DATE_RE, MONEY_RE, repair_money_ocr
from models.total_post import extract_total_value

try:
    import torch
    from torch import Tensor as _Tensor  # noqa: F401  (silence ruff SIM105)
except ImportError:  # lightweight CI — torch not installed
    pass

if TYPE_CHECKING:
    import torch


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
    # PR #113 / H1 — only request ``kv`` when the FOCUS-A span head is
    # present; the no-span path stays bit-identical to the legacy
    # ``assigner(...)`` call.
    kv: torch.Tensor | None
    if assigner._span_head is not None:
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
    )
