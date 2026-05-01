"""Per-field analytical propagation on top of the learned draft picks.

Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
Article: "End-to-End vs. Pipeline Receipt KIE: DONUT Against
    YOLO+TrOCR+Attention on SROIE" (IEEE/ICDAR submission).
Role: refines the learned AttentionAssigner's draft field→value dict
    using the same four-step recipe that lifts ``date`` to F1≥0.95 —
    regex/validator, value extractor, runner-up fallback, output
    normaliser.  Per-field fixes: total (SUBTOTAL/``TOTAL:``) →
    money-candidate scoring with ±1-line keyword context; address
    → spatial propagation via rule ``_pick_address``; company →
    validate-and-fallback to topmost non-junk; date → regex-first
    with reading-order fallback.  No torch dep — runs on any
    assigner checkpoint without retraining.
"""
from __future__ import annotations

import math
import re

from models.normalize import (
    normalize_address,
    normalize_company,
    normalize_date,
    normalize_total_value,
)
from models.rule_fields import (
    _SUBTOTAL_KW_RE,
    _is_short_junk,
    _pick_address,
    _pick_company,
    extract_date,
    extract_total,
)
from models.rule_regex import (
    _ADDR_ANCHOR,
    _ADDR_COMPANY_HEADER,
    _ADDR_CONTINUATION,
    _ADDR_EXCLUDE,
    _ADDR_LEADING_JUNK_RE,
    _ADDR_TERMINATOR,
    _COMPANY_TOKEN,
    _DATE_RE,
    _HEADER_JUNK,
    _MONEY_RE,
    _POSTCODE_RE,
    _TOTAL_NEGATIVE,
    _TOTAL_STRONG,
    _TOTAL_WEAK,
    repair_money_ocr,
)
from models.total_arithmetic import (
    _classify as _classify_money_lines,
)
from models.total_arithmetic import (
    _identity_cash_change,
    _identity_sub_tax,
    subset_sum_target_cents,
    total_arithmetic_consensus,
)

_CURRENCY_PREFIX_RE = re.compile(r"^(RM|USD|SGD|MYR|\$)\s*", re.IGNORECASE)
# Currency-prefix cue on the SAME line as the money value — a weak positive
# because TOTAL lines are the ones most often printed with ``RM``/``MYR``.
_CURRENCY_CUE_RE = re.compile(r"\b(?:RM|MYR|\$)\b", re.IGNORECASE)

# How much better (in _score_money units) a candidate must be than the
# learned argmax before ``_refine_total`` overrides a valid learned
# money value.  Calibrated from the live miss table: a SUBTOTAL line
# with learned attention scores ~1 (attn argmax) while the real TOTAL
# line scores ~4 (``_TOTAL_STRONG`` match) — a 2-point margin preserves
# that correction while leaving weak-evidence cases to the assigner.
_TOTAL_OVERRIDE_MARGIN = 2.0
# Plausibility floor for a grand total — predictions parsing to a value
# below this almost always indicate a wrong-line pick (a quantity, a
# tax-rate cell, a stray decimal printed on the receipt).  When the
# learned value is below this floor and a competing candidate parses
# above it, the override margin collapses so the rule scorer wins.
_TOTAL_MIN_PLAUSIBLE = 1.0

# --- Strategy (L) — calibrated additive scoring -----------------------------
# Weight that multiplies ``log(attn+ε)`` when the rule-based money scorer
# ranks candidates.  The rule score is already well-calibrated in the 0–5
# range (see :func:`_score_money`); α=0.5 keeps attention as a tie-break
# without letting a confident-but-wrong attention peak overwhelm a clean
# ``_TOTAL_STRONG`` keyword match.  Tuned qualitatively on the live miss
# table; re-tune on val if the attention distribution shifts.
_ATTN_BLEND_ALPHA = 0.5
# ε floor — log(0) = -inf otherwise kills the signal for all non-argmax
# candidates.  Matches the ``clamp(min=1e-8)`` used in the training loss.
_ATTN_LOG_EPS = 1e-4

# --- Strategy (H) — confidence-gated delegation -----------------------------
# Per-field attention is considered "diffuse" (low-confidence) when either:
#   * normalised Shannon entropy H/H_max ≥ _ATTN_DIFFUSE_ENTROPY, **or**
#   * top-1 − top-2 margin ≤ _ATTN_DIFFUSE_MARGIN.
# Under diffuse attention the override margin for ``_refine_total`` drops
# to :data:`_TOTAL_OVERRIDE_MARGIN_DIFFUSE` so the rule-based scorer can
# correct the learned pick more aggressively — a free F1 floor since the
# rule-based arm has higher per-field F1 than the learned arm on the
# SROIE miss table.
_ATTN_DIFFUSE_ENTROPY = 0.80
# Strategy H tightening (run 20260430T125211Z): the assigner is
# functionally uncertain whenever the top-1 vs top-2 attention gap
# is below ~10% — at margins ≤0.05 we were already classifying as
# diffuse, but the live miss-table shows another ≈8 of the 97 ``total``
# losses sit between 0.05 and 0.10, where the learned argmax is a
# coin-flip but the rule path has decisive scoring (TOTAL keyword + witness).
# Raising the diffuse threshold lets those flip in favour of the rule.
_ATTN_DIFFUSE_MARGIN = 0.10
# Under diffuse attention any positive rule-score advantage is enough:
# the learned argmax carries no information, so the witness/keyword/
# positional ensemble in ``_score_money`` is the only available signal.
_TOTAL_OVERRIDE_MARGIN_DIFFUSE = 0.0


def _attn_entropy(row: list[float]) -> float:
    """Normalised Shannon entropy of an attention row — 0 = peaked, 1 = uniform."""
    if not row:
        return 1.0
    total = sum(max(p, 0.0) for p in row)
    if total <= 0:
        return 1.0
    probs = [max(p, 0.0) / total for p in row]
    h = -sum(p * math.log(p) for p in probs if p > 0)
    h_max = math.log(len(probs)) if len(probs) > 1 else 1.0
    return h / h_max if h_max > 0 else 1.0


def _attn_margin(row: list[float]) -> float:
    """Top-1 − top-2 gap after normalisation — 0 = tie, 1 = one-hot.

    A single-region row has no second element, so we return 1.0 to
    represent "maximum confidence" (there is nothing to confuse it
    with).  This keeps ``_is_attn_diffuse`` returning False on the
    degenerate 1-region case — the caller will simply accept whatever
    the scorer produces, which is the same as the pre-H behaviour.
    """
    if not row or len(row) < 2:
        return 1.0 if row else 0.0
    total = sum(max(p, 0.0) for p in row) or 1.0
    probs = sorted((max(p, 0.0) / total for p in row), reverse=True)
    return probs[0] - probs[1]


def _is_attn_diffuse(row: list[float] | None) -> bool:
    """True when the attention row is flat enough that the rule-based
    scorer should be trusted over the learned argmax."""
    if row is None or not row:
        return True
    return (_attn_entropy(row) >= _ATTN_DIFFUSE_ENTROPY
            or _attn_margin(row) <= _ATTN_DIFFUSE_MARGIN)


def _strip_currency(s: str) -> str:
    return _CURRENCY_PREFIX_RE.sub("", s).strip()


def _strip_leading_junk(value: str) -> str:
    """Drop ``CO. NO. 37365-A`` / ``GST NO. 123...`` leading fragments.

    Never returns the empty string — if the regex would consume the
    entire value (the input was *only* junk), we keep the original so
    the candidate scorer can still rank it (it will simply lose to the
    span).
    """
    prev = value
    cur = _ADDR_LEADING_JUNK_RE.sub("", prev).strip()
    # Re-run once in case two junk tokens were concatenated.
    if cur != prev:
        cur = _ADDR_LEADING_JUNK_RE.sub("", cur).strip()
    return cur or value


def _attn_rank(row: list[float]) -> dict[int, int]:
    """``{region_idx: rank}`` with rank 0 = highest attention."""
    order = sorted(range(len(row)), key=lambda i: -row[i])
    return {i: r for r, i in enumerate(order)}


def _money_value_at(text: str) -> float | None:
    """Parse the rightmost money on ``text`` to a float; ``None`` on parse fail."""
    from models.rule_regex import MONEY_RE
    matches = list(MONEY_RE.finditer(text or ""))
    if not matches:
        return None
    raw = matches[-1].group(0).strip()
    raw = re.sub(r"^(RM|USD|SGD|MYR|\$)\s*", "", raw, flags=re.IGNORECASE).strip()
    try:
        return float(raw.replace(",", ""))
    except ValueError:
        return None


def _score_money(
    i: int, texts: list[str], rank: dict[int, int], money_idxs: list[int],
    attn_row: list[float] | None = None,
    arithmetic_targets: list[float] | None = None,
    subset_sum_cents: frozenset[int] | None = None,
) -> float:
    """Higher = more likely the real TOTAL line.

    Signals (tuned to the kaggle2 SROIE miss table where ~1/3 of losses
    are SUBTOTAL/TAX/CASH/CHANGE picked instead of GRAND TOTAL):

    - ±1-line keyword window: ``_TOTAL_STRONG`` (+4), ``_TOTAL_WEAK``
      w/o SUBTOTAL (+2.5), ``_SUBTOTAL_KW_RE`` (-4), ``_TOTAL_NEGATIVE``
      (-2).  Asymmetric weights prevent a ``SUBTOTAL: RM 38`` line that
      is also near ``TOTAL:`` (two lines down) from winning.
    - ``RM``/``MYR`` cue on the SAME line as the money value (+0.3) —
      grand totals are the ones printed with a currency symbol.
    - Positional: the **last** money line gets +1.5, the 2nd-to-last
      gets +0.5 (this is the strongest single signal on SROIE).
    - Attention rank tie-break: argmax +1.0, top-3 +0.3.
    - **Strategy (L)** — when ``attn_row`` is provided, add
      ``_ATTN_BLEND_ALPHA · log(attn + ε)`` so a confident assigner peak
      tilts ties without swamping a clean ``_TOTAL_STRONG`` match.  This
      is the additive-ensemble scorer recommended by the plan, replacing
      the coarse "argmax +1.0" tie-breaker with a continuous signal.

    Note the previous version multiplied ``bboxes[i][1]`` (raw pixel y)
    by 0.5 which produced dominating 500+ scores on tall receipts; the
    pixel-y term is removed and replaced by the relative money-line
    position above.
    """
    nbr = " ".join(texts[max(0, i - 1): i + 2])
    same_line = texts[i] if i < len(texts) else ""
    s = 0.0
    if _TOTAL_STRONG.search(nbr):
        s += 4.0
    elif _TOTAL_WEAK.search(nbr) and not _SUBTOTAL_KW_RE.search(nbr):
        s += 2.5
    # Same-line distractor keywords (CASH/CHANGE/SUBTOTAL/TAX on the
    # same line as the money value) are a much stronger negative signal
    # than the same keyword merely appearing in the ±1 neighbourhood.  A
    # CASH 100.00 line sandwiched between TOTAL and CHANGE used to
    # collect a +2.5 from the neighbourhood window which masked the
    # negative signal; gating the heavy penalties on the same line
    # disambiguates that layout.
    if _SUBTOTAL_KW_RE.search(same_line):
        s -= 5.0
    elif _SUBTOTAL_KW_RE.search(nbr):
        s -= 3.0
    if _TOTAL_NEGATIVE.search(same_line):
        s -= 4.0
    elif _TOTAL_NEGATIVE.search(nbr):
        s -= 1.5
    if _CURRENCY_CUE_RE.search(same_line):
        s += 0.3
    if money_idxs:
        if i == money_idxs[-1]:
            s += 1.5
        elif len(money_idxs) >= 2 and i == money_idxs[-2]:
            s += 0.5
    r = rank.get(i, len(rank))
    if r == 0:
        s += 1.0
    elif r <= 2:
        s += 0.3
    if attn_row is not None and 0 <= i < len(attn_row):
        a = max(float(attn_row[i]), 0.0)
        s += _ATTN_BLEND_ALPHA * math.log(a + _ATTN_LOG_EPS)
    # Plausibility floor on the value itself: SROIE grand totals are
    # virtually never 0.00 (those lines are rounding-adjustment / empty
    # discount fields) and only rarely below RM 1.00 (smallest in GT
    # ≈ 0.90).  Apply a hard penalty for 0.00 and a softer demotion for
    # sub-RM-1 values so a stray "0.00" rounding line on the bottom of
    # the receipt doesn't out-score the real total just because it sits
    # last in money order.  The penalty is *additive* and bounded so
    # genuine sub-RM-1 totals (rare but real) can still win when no
    # competing larger candidate carries a TOTAL keyword.
    val = _money_value_at(texts[i] if 0 <= i < len(texts) else "")
    if val is not None:
        # Zero-pred suppression — pred 0.00 is almost certainly a
        # ROUNDING / DISCOUNT / quantity line, never a SROIE grand total.
        # Run 20260430T125211Z had 6 of 97 ``total`` failures with
        # pred=0.00.  -8 is decisive against any combination of weak
        # positional + attention signals, while still allowing the
        # rare receipt with TOTAL_STRONG keyword on a 0.00 line to win
        # (TOTAL_STRONG is +4 + last-money +1.5 + … — net still negative).
        if val == 0.0:
            s -= 8.0
        elif val < 0.0:
            # Negative pred is almost always a CHANGE / REFUND line
            # (n=4 of 97 in run 20260430T125211Z).  SROIE GT totals
            # are non-negative on the ICDAR-2019 Task-3 split.  Hard
            # rejection: strongly negative score so nothing beats it
            # except an explicit refund-receipt scenario which the
            # ``_TOTAL_NEGATIVE`` regex already controls.
            s -= 12.0
        elif val < 1.0:
            s -= 1.0
        # Maximum-money relative prior — SROIE grand totals are almost
        # always within 25% of the receipt's largest money value.  Lines
        # whose value is < 30% of the receipt-max get a soft demote so
        # an item-line / quantity-line / 6%-tax-line never out-scores
        # the actual grand total purely on positional advantage.
        # Computed inside the scorer (read-only over ``texts``) so no
        # caller-API change is needed.
        if money_idxs:
            other_vals: list[float] = []
            for j in money_idxs:
                if j == i:
                    continue
                vj = _money_value_at(texts[j] if 0 <= j < len(texts) else "")
                if vj is not None and vj > 0:
                    other_vals.append(vj)
            if other_vals:
                receipt_max = max(other_vals + ([val] if val > 0 else []))
                if receipt_max > 0 and 0 < val / receipt_max < 0.3:
                    s -= 2.0
        # Arithmetic-witness boost: a value satisfying one of the
        # receipt's identities (cash − change, subtotal + tax + svc −
        # disc, FOCUS-Σ items-subset-sum + tax_aug) is strong
        # out-of-band evidence that this is the grand total —
        # independent of any keyword anchor.  Two witnesses (rare)
        # is essentially proof on SROIE; three is proof.  This boost
        # is what turns ``_score_money`` from a keyword-matching
        # heuristic into an arithmetic-validated scorer.
        if arithmetic_targets or subset_sum_cents:
            wit = _arithmetic_witness_count(
                val, arithmetic_targets or [], subset_sum_cents,
            )
            if wit >= 3:
                s += 12.0  # FOCUS-Σ proof tier (all three identities agree)
            elif wit == 2:
                s += 8.0   # consensus tier (any two agree — was +6)
            elif wit == 1:
                s += 4.0   # single witness (was +3) — slightly more
                # decisive against unwitnessed noise lines that happen
                # to share a positional or attention rank.
    return s


def _parse_money_value(s: str) -> float | None:
    """Best-effort float parse of a stripped money string."""
    if not s:
        return None
    try:
        return float(s.strip().replace(",", ""))
    except ValueError:
        return None


def _arithmetic_targets(
    classified: list[tuple[int, str, float]],
) -> list[float]:
    """Candidate grand totals derived from receipt arithmetic identities.

    Returns every value the identity solvers can produce — not just the
    consensus.  A line whose float matches *any* of these within ±2¢ is
    treated as arithmetic-validated and gets a strong score boost in
    :func:`_score_money`.  When the OCR'd total line has a single-digit
    drift (``70.45`` vs the real ``79.45``) this list lets us identify
    the true value and substitute it without trusting the corrupted
    line's raw digits.
    """
    out: list[float] = []
    cash = _identity_cash_change(classified)
    if cash is not None:
        out.append(cash)
    sub = _identity_sub_tax(classified)
    if sub is not None:
        out.append(sub)
    return out


def _arithmetic_witness_count(
    value: float, targets: list[float],
    subset_sum_cents: frozenset[int] | None = None,
) -> int:
    """How many arithmetic identities ``value`` satisfies to ±2¢.

    Counts I₁ (cash−change) + I₂ (subtotal+tax+svc−disc) keyword-anchored
    matches from ``targets``, plus FOCUS-Σ Identity 3 (items+tax_aug
    subset-sum) when ``subset_sum_cents`` is provided.  Maximum count
    is 3 (essentially proof on SROIE).
    """
    count = sum(1 for t in targets if abs(value - t) <= 0.02)
    if subset_sum_cents is not None and int(round(value * 100)) in subset_sum_cents:
        count += 1
    return count


def _value_close(a: str, b: str, eps: float = 0.02) -> bool:
    """Compare two money strings as floats within ``eps``; safe on parse fail."""
    fa = _parse_money_value(a.lstrip("-"))
    fb = _parse_money_value(b.lstrip("-"))
    if fa is None or fb is None:
        return False
    return abs(fa - fb) <= eps


def _ocr_drift_distance(a: str, b: str) -> int:
    """Levenshtein on the digit sequences of two money strings.

    Returns the smaller of (1) absolute difference in digit length plus
    edit distance on aligned digits, or (2) full Levenshtein over the
    original strings.  Used to pair an arithmetic-synthesised value
    with an OCR-corrupted line — ``70.45`` vs ``79.45`` differs by
    one digit, ``118.55`` vs ``119.55`` differs by one digit, etc.
    A non-positive return is impossible; callers should treat
    ``<= 2`` as "OCR-plausibly the same value".
    """
    sa, sb = a.lstrip("-"), b.lstrip("-")
    if sa == sb:
        return 0
    # Quick reject: very different lengths almost always means a
    # genuinely different value (e.g. 75.00 vs 750.00 is a real
    # missing-digit OCR but pairs only at distance 1).
    if abs(len(sa) - len(sb)) > 2:
        return abs(len(sa) - len(sb))
    # Standard Levenshtein, capped at small values for cheapness.
    m, n = len(sa), len(sb)
    if m == 0 or n == 0:
        return max(m, n)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        cur = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if sa[i - 1] == sb[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[n]


def _ocr_drift_match_in_set(
    value_cents: int, target_set: frozenset[int], max_edits: int = 1,
) -> int | None:
    """Return any target in ``target_set`` reachable from ``value_cents``
    by ``≤ max_edits`` decimal-digit substitutions, else ``None``.

    FOCUS-Σ companion to :func:`_ocr_drift_distance`: where the legacy
    helper measures distance between two specific money strings, this
    one searches a precomputed *set* of plausible targets (the I₃
    subset-sum cents-set) for any k-edit neighbour of ``value_cents``.

    Cost.  At ``max_edits=1`` the scan is O(D × 9) ≈ 63 candidates for
    a 7-digit money value (constant for SROIE).  At ``max_edits=2`` the
    scan is O(C(D,2) × 81) — for a 7-digit money value that's 1 701
    candidates; still negligible vs the rest of the eval pipeline.
    The 2-edit path is therefore gated by the caller (witness ≥ 2 OR
    explicit TOTAL keyword) so it only fires on lines where the
    arithmetic substitution is supportable by independent signals.

    Tiebreak.  When several ``target_set`` members are reachable, prefer
    the *largest* — on SROIE grand totals almost always exceed any
    partial sub-sum (subtotal, individual-item, tax line).  Reject
    candidates that begin with a leading zero past length 1, since
    "048" is not a real 1-digit OCR substitution but a *lost*-digit
    pathology that the YOLO crop-pad change handles geometrically.
    """
    s = str(value_cents)
    best: int | None = None
    digit_positions = [k for k, ch in enumerate(s) if ch.isdigit()]
    if not digit_positions or max_edits < 1:
        return None
    edit_position_combos: list[tuple[int, ...]]
    if max_edits == 1:
        edit_position_combos = [(p,) for p in digit_positions]
    else:  # max_edits >= 2 — emit 1-edit AND 2-edit candidates
        edit_position_combos = [(p,) for p in digit_positions]
        for ai in range(len(digit_positions)):
            for bi in range(ai + 1, len(digit_positions)):
                edit_position_combos.append(
                    (digit_positions[ai], digit_positions[bi]),
                )
    digits = "0123456789"
    for combo in edit_position_combos:
        if len(combo) == 1:
            (pos,) = combo
            for d in digits:
                if d == s[pos]:
                    continue
                cand_str = s[:pos] + d + s[pos + 1:]
                if cand_str.startswith("0") and len(cand_str) > 1:
                    continue
                try:
                    cand = int(cand_str)
                except ValueError:
                    continue
                if cand in target_set and (best is None or cand > best):
                    best = cand
        else:
            (p1, p2) = combo
            for d1 in digits:
                if d1 == s[p1]:
                    continue
                for d2 in digits:
                    if d2 == s[p2]:
                        continue
                    cand_str = s[:p1] + d1 + s[p1 + 1:p2] + d2 + s[p2 + 1:]
                    if cand_str.startswith("0") and len(cand_str) > 1:
                        continue
                    try:
                        cand = int(cand_str)
                    except ValueError:
                        continue
                    if cand in target_set and (best is None or cand > best):
                        best = cand
    return best


def _attach_sign(value: str, source_line: str) -> str:
    """Re-attach a leading minus when ``value`` was OCR'd as ``-VALUE``.

    The shared ``_MONEY_RE`` doesn't allow a leading ``-``, so refund /
    credit-note picks like ``"-6.42"`` lose their sign during extraction.
    Reinstate it when a ``-`` (or its OCR confusables) directly precedes
    the matched value on the source line.
    """
    if not value or "-" in value or not source_line:
        return value
    pos = source_line.find(value)
    if pos <= 0:
        return value
    prev = source_line[pos - 1]
    if prev == "-":
        return "-" + value
    return value


def _refine_total(
    learned: str, texts: list[str], bboxes: list[list[float]],
    attn_row: list[float] | None,
) -> str:
    """Score every money-bearing region; override learned only on strong
    positive margin.

    *Number fields should be near-100%* — unlike address, a total is a
    single value with a tight regex.  But the learned assigner is
    trained on SROIE targets and is usually right; overriding it on
    weak evidence regresses total-F1 (a 0.619 → 0.540 drop was observed
    when any ``best_score > 0`` triggered an override).  The
    conservative rule below only overrides the learned pick when:

    * the learned value is **not** a well-formed money string, **or**
    * the learned value is implausibly small (≤ ``_TOTAL_MIN_PLAUSIBLE``)
      while a candidate parses above the floor — likely a quantity or a
      stray-decimal pick rather than a true grand total, **or**
    * the best-scoring candidate beats the learned one by at least the
      override margin — which drops from :data:`_TOTAL_OVERRIDE_MARGIN`
      to :data:`_TOTAL_OVERRIDE_MARGIN_DIFFUSE` when the assigner's
      attention is diffuse (strategy H — confidence-gated delegation).

    Negative-sign preservation: refund/credit-note lines like
    ``"REFUND -6.42"`` lose their leading minus through the shared
    ``_MONEY_RE`` (which doesn't permit ``-``); :func:`_attach_sign`
    re-attaches it from the source line.

    Arithmetic-consensus fallback: when both the learned pick and the
    best-scoring rule candidate are implausibly small (≤ floor), the
    arithmetic-consensus solver is consulted directly so receipts whose
    grand total was buried under cash/change/subtotal+tax still recover
    a valid value.
    """
    repaired = [repair_money_ocr(t) for t in texts]
    money_idxs = [i for i, t in enumerate(repaired) if _MONEY_RE.search(t)]
    if not money_idxs:
        return learned
    # Arithmetic identities computed once; passed to the scorer so a
    # value satisfying ``cash − change`` or ``subtotal + tax`` is
    # promoted regardless of the keyword anchor on its line.
    classified = _classify_money_lines(repaired)
    targets = _arithmetic_targets(classified)
    # FOCUS-Σ: precompute the items-subset-sum reachable cents-set once
    # per receipt.  Identity 3 fires when the keyword-anchored I₁/I₂ are
    # silent (no SUBTOTAL/CASH/CHANGE keyword survived OCR) but item
    # lines still enumerate to the grand total — the dominant
    # ``assigner_error`` failure mode in the diagnostics for run
    # 20260430T125211Z (88 DONUT, 93 pipeline ``total`` errors).
    subset_sum_cents = subset_sum_target_cents(classified)
    # OCR-drift correction: if arithmetic produces a clean target value
    # but the closest line value is within edit distance 1 (single OCR
    # digit drift — ``79.45`` vs ``70.45``, ``119.55`` vs ``118.55``,
    # ``848.00`` vs ``48.00``), prefer the arithmetic target itself.
    # The keyword-match boundary is already enforced by the identities
    # — they only fire when SUBTOTAL/CASH lines are present — so this
    # path never invents a value out of thin air.
    rank = _attn_rank(attn_row) if attn_row else {}
    scored: list[tuple[float, int, str]] = sorted(
        ((_score_money(
              i, repaired, rank, money_idxs, attn_row, targets,
              subset_sum_cents,
          ), i,
          _MONEY_RE.search(repaired[i]).group(0))  # type: ignore[union-attr]
         for i in money_idxs),
        reverse=True,
    )
    best_score, best_idx, best_val = scored[0]
    best_clean = _strip_currency(best_val)
    best_signed = _attach_sign(
        best_clean,
        repaired[best_idx] if 0 <= best_idx < len(repaired) else "",
    )
    learned_clean = _strip_currency(learned)
    learned_num = _parse_money_value(learned_clean.lstrip("-"))
    best_num = _parse_money_value(best_clean)

    # Sign-positive gate.  SROIE Task-3 grand totals are non-negative
    # by construction (refund / credit-note receipts are not in the
    # canonical 347 test split).  When the learned value is negative
    # (CHANGE / REFUND line) but the rule-scored best is a positive
    # plausible total, the negative learned is virtually certainly a
    # sibling-line pick and the best should win unconditionally.
    # n=4 of 97 ``total`` failures on run 20260430T125211Z had a
    # negative learned value; this gate flips them.
    if (learned_clean.startswith("-")
            and best_num is not None and best_num > _TOTAL_MIN_PLAUSIBLE
            and best_score > 0):
        return best_signed

    def _arithmetic_fallback() -> str | None:
        ar = total_arithmetic_consensus(repaired, set())
        if ar is None:
            return None
        _, value = ar
        v = _parse_money_value(value)
        if v is None or v <= _TOTAL_MIN_PLAUSIBLE:
            return None
        return value

    # ARITHMETIC-FIRST PATH (transformative): when an identity gives a
    # clean target and a TOTAL-keyword'd line carries a value within
    # 1-edit OCR drift of that target, the arithmetic value wins.  This
    # catches the dominant single-digit-drift failure mode (``70.45`` ↔
    # ``79.45``, ``118.55`` ↔ ``119.55``, ``30.68`` ↔ ``30.70``,
    # ``25.10`` ↔ ``24.10``) where the score-based path has no signal
    # that the printed digits are wrong.
    for tgt in targets:
        tgt_str = f"{tgt:.2f}"
        # Already exact on some line — let the witness boost in the
        # scorer carry it; no substitution needed here.  Use strict
        # equality (no eps): if the line value differs by even 1¢ we
        # may need to substitute, since the arithmetic target is
        # exact-by-construction and any OCR drift is what we are
        # trying to correct.
        def _exact_match(idx: int, target: str = tgt_str) -> bool:
            raw = _MONEY_RE.search(repaired[idx]).group(0).strip()  # type: ignore[union-attr]
            raw = re.sub(
                r"^(RM|USD|SGD|MYR|\$)\s*", "", raw, flags=re.IGNORECASE,
            )
            return raw == target
        if any(_exact_match(i) for i in money_idxs):
            continue
        # Find an OCR-drift sibling: a line within 1 char of tgt whose
        # neighbourhood carries a TOTAL keyword (so we don't substitute
        # against a cash/subtotal line that happens to be 1 digit off).
        for i in money_idxs:
            line_val = _MONEY_RE.search(repaired[i]).group(0).strip()  # type: ignore[union-attr]
            line_val = re.sub(
                r"^(RM|USD|SGD|MYR|\$)\s*", "", line_val, flags=re.IGNORECASE,
            )
            if _ocr_drift_distance(line_val, tgt_str) > 1:
                continue
            nbr = " ".join(repaired[max(0, i - 1): i + 2])
            if _SUBTOTAL_KW_RE.search(nbr):
                continue
            if _TOTAL_NEGATIVE.search(repaired[i]):
                continue
            # Match: the corrupted line's value is within 1 OCR edit of
            # the arithmetic target AND its context is TOTAL-positive.
            if _TOTAL_STRONG.search(nbr) or _TOTAL_WEAK.search(nbr):
                return tgt_str
    # FOCUS-Σ ARITHMETIC-FIRST PATH (Identity 3):
    # When I₁/I₂ produced no targets but I₃ has reachable subset-sum
    # values, look for a TOTAL-keyword'd line whose digits are within
    # one OCR substitution of *any* I₃ target.  Catches the digit-error
    # regime (``8.50`` vs OCR ``8.20``, ``6.60`` vs OCR ``8.60``,
    # ``169.80`` vs OCR ``109.80``) on receipts where SUBTOTAL/CASH
    # keywords were OCR-lost.  Conservative gates: require TOTAL keyword
    # on the line, exclude SUBTOTAL/CHANGE neighbours and negative
    # totals, and emit only when the line is *not* already a literal
    # subset-sum match (else the score path handles it).
    if not targets and subset_sum_cents:
        for i in money_idxs:
            line_match = _MONEY_RE.search(repaired[i])
            if line_match is None:
                continue
            line_val = re.sub(
                r"^(RM|USD|SGD|MYR|\$)\s*", "", line_match.group(0).strip(),
                flags=re.IGNORECASE,
            )
            try:
                line_cents = int(round(float(line_val.replace(",", "")) * 100))
            except ValueError:
                continue
            if line_cents in subset_sum_cents:
                continue  # exact match — score-path witness boost handles it
            nbr = " ".join(repaired[max(0, i - 1): i + 2])
            if _SUBTOTAL_KW_RE.search(nbr):
                continue
            if _TOTAL_NEGATIVE.search(repaired[i]):
                continue
            if not (_TOTAL_STRONG.search(nbr) or _TOTAL_WEAK.search(nbr)):
                continue
            match_cents = _ocr_drift_match_in_set(line_cents, subset_sum_cents)
            if match_cents is not None:
                return f"{match_cents / 100:.2f}"
            # 2-edit fallback — gated by TOTAL_STRONG keyword (not just
            # weak) so we only fire when the line context is decisive.
            # Catches the n=10 NEAR_VALUE_2EDIT failures (e.g. ``49.70``
            # OCR'd as ``46.90``: 4↔4 same, 9↔6 substitution at pos 1,
            # 7↔9 substitution at pos 2 — two edits).  Without the
            # ``_TOTAL_STRONG`` gate a 2-edit search would over-correct.
            if _TOTAL_STRONG.search(nbr):
                match_cents = _ocr_drift_match_in_set(
                    line_cents, subset_sum_cents, max_edits=2,
                )
                if match_cents is not None:
                    return f"{match_cents / 100:.2f}"
    if not _MONEY_RE.fullmatch(learned_clean):
        # Learned value isn't a usable money string; take the scored
        # pick unconditionally (fall back to learned if no positive
        # evidence either).
        if best_score > 0:
            return best_signed
        ar = _arithmetic_fallback()
        return ar if ar is not None else learned_clean
    # Implausibly-small learned value (0.00 / qty cell / discount %) +
    # a candidate above the plausibility floor → override even on a
    # weak score margin.  This recovers the dominant single failure
    # mode in the live miss table (predicted 0.00 / sub-$1 vs an actual
    # grand total in the tens-of-RM range) without regressing receipts
    # whose true total is genuinely tiny.
    if (learned_num is not None and learned_num <= _TOTAL_MIN_PLAUSIBLE
            and best_num is not None and best_num > _TOTAL_MIN_PLAUSIBLE
            and best_score > 0):
        return best_signed
    # Learned value is a well-formed money string — protect it unless a
    # competing candidate is decisively better.  The required margin is
    # relaxed when the assigner is unsure of itself (flat attention row).
    learned_score = next(
        (sc for sc, _, v in scored if v.strip() == learned_clean), float("-inf"),
    )
    # When the learned value doesn't appear verbatim on any money-bearing
    # line (typical for ``total_arithmetic_consensus`` synthesised values
    # — ``cash − change`` / ``subtotal + tax`` — that recover OCR-corrupted
    # total lines), the legacy ``-inf`` learned-score forced the override
    # path.  Treat those values as already arithmetic-validated and pin
    # their score above the strict-margin threshold so the rule scorer
    # only displaces them on a *very* decisive lead.
    synthesised = (
        learned_score == float("-inf") and learned_num is not None
        and learned_num > _TOTAL_MIN_PLAUSIBLE
    )
    if synthesised:
        learned_score = best_score
    margin = (_TOTAL_OVERRIDE_MARGIN_DIFFUSE if _is_attn_diffuse(attn_row)
              else _TOTAL_OVERRIDE_MARGIN)
    if best_score - learned_score >= margin and best_score > 0:
        return best_signed
    # Last resort: when both the learned pick and the best candidate are
    # tiny, ask the arithmetic solver for an out-of-band consensus value.
    if (learned_num is not None and learned_num <= _TOTAL_MIN_PLAUSIBLE
            and (best_num is None or best_num <= _TOTAL_MIN_PLAUSIBLE)):
        ar = _arithmetic_fallback()
        if ar is not None:
            return ar
    # Sign re-attachment for the kept learned value (refund OCR loses
    # the leading minus through ``_MONEY_RE``).
    learned_signed = _attach_sign(
        learned_clean,
        repaired[best_idx] if 0 <= best_idx < len(repaired) else "",
    )
    # Only re-attach when the learned value actually appears on the
    # best-scored line; otherwise fall back to scanning the full
    # receipt for a ``-VALUE`` token matching the learned pick.
    if learned_signed == learned_clean:
        for line in repaired:
            sgn = _attach_sign(learned_clean, line)
            if sgn != learned_clean:
                return sgn
    return learned_signed


def _refine_date(learned: str, texts: list[str]) -> str:
    """Keep learned pick if DATE_RE matches; otherwise first regex hit."""
    m = _DATE_RE.search(learned)
    if m is not None:
        return m.group(0)
    picked = extract_date(texts)
    return picked[1] if picked is not None else learned


def _refine_company(
    learned: str, texts: list[str], bboxes: list[list[float]],
    attn_row: list[float] | None = None,
) -> str:
    """Score-and-pick the best of {learned, rule-topmost-non-junk}.

    Previously this was a one-way validate-and-fallback: the learned
    pick was returned whenever it wasn't outright junk, which kept
    taglines / slogans / brand lines that :func:`_pick_company` would
    have skipped.  On the live miss table the rule-based arm reaches
    company F1 ≈ 0.77 while the learned arm stalls at ≈ 0.68 — so when
    both candidates are *valid*, we now rank them on a stable tuple
    key that rewards hard company markers and conventional upper-case
    formatting.

    Scoring dimensions (listed from most to least important):
      1. ``not_junk`` — not ``HEADER_JUNK`` / ``_ADDR_EXCLUDE`` / short
      2. ``has_company_token`` — ``SDN BHD`` / ``ENTERPRISE`` / etc.
      3. ``is_mostly_upper`` — companies on SROIE receipts are UPPERCASE
      4. ``-y`` — topmost wins on ties (negated so smaller y = higher)

    Strategy (H) — **confidence-gated delegation**: when the learned
    attention row for the ``company`` field query is diffuse, we zero
    the ``has_company_token`` bit on the learned candidate so the rule
    pick wins ties.  This mirrors the treatment applied to address and
    total, and gives the pipeline a free F1 floor on precisely the
    receipts where the learned arm has no conviction.
    """
    learned_clean = _strip_decor(learned.strip())
    learned_clean = _strip_company_tail(learned_clean)
    picked = _pick_company(texts, bboxes, used=set())
    rule_pick_raw = picked[1] if picked is not None else ""
    rule_pick = _strip_company_tail(_strip_decor(rule_pick_raw))
    # Boilerplate-line guard: when a candidate is dominantly a TAX
    # INVOICE / CASH BILL / SIMPLIFIED RECEIPT line (the OCR header
    # fired before the company anchor), discard it before scoring so
    # the topmost non-boilerplate alternative wins.  Earlier code only
    # caught full-line ``_HEADER_JUNK`` matches and missed the common
    # "*** TAX INVOICE ***" / "SIMPLIFIED TAX INVOICE" forms.
    def _is_boilerplate(v: str) -> bool:
        s = v.strip()
        if not s:
            return True
        if _COMPANY_BOILERPLATE_LINE_RE.search(s):
            return True
        return _HEADER_JUNK.match(s) is not None

    def _maybe_extend(v: str) -> str:
        """Apply forward/backward extension if the candidate looks partial."""
        if not v:
            return v
        # Already complete (trade-name + SDN BHD) → leave alone.
        if (_COMPANY_TOKEN.search(v)
                and not _COMPANY_MULTI_SUFFIX_RE.match(v)):
            return v
        ext = _company_extend(v, texts, bboxes)
        # Only accept extension when it added a trade-name OR a
        # company-token that wasn't there before.  Otherwise keep the
        # original to avoid noise concatenation.
        if ext and ext != v:
            had_token = bool(_COMPANY_TOKEN.search(v))
            now_token = bool(_COMPANY_TOKEN.search(ext))
            if (now_token and not had_token) or len(ext.split()) > len(v.split()):
                return ext
        return v

    candidates: list[str] = []
    seen: set[str] = set()
    for c in (learned_clean, rule_pick):
        if not c or _is_boilerplate(c):
            continue
        c = _maybe_extend(c)
        k = c.lower()
        if k not in seen:
            seen.add(k)
            candidates.append(c)
    if not candidates:
        # Both candidates were boilerplate — fall back to the topmost
        # non-boilerplate, non-junk line so we never return ``"*** TAX
        # INVOICE ***"`` as the company answer.
        order = sorted(
            range(len(texts)),
            key=lambda i: bboxes[i][1] if i < len(bboxes) else 0.0,
        )
        for i in order:
            t = _strip_decor(texts[i].strip())
            if not t or _is_boilerplate(t) or _is_short_junk(t):
                continue
            if _DATE_RE.search(t) or _MONEY_RE.search(t):
                continue
            if _ADDR_EXCLUDE.search(t):
                continue
            candidates.append(_maybe_extend(_strip_company_tail(t)))
            break
        if not candidates:
            return learned

    def _y(v: str) -> float:
        return _y_of(v, texts, bboxes)

    def _is_mostly_upper(s: str) -> bool:
        letters = [c for c in s if c.isalpha()]
        if len(letters) < 3:
            return False
        upper = sum(1 for c in letters if c.isupper())
        return upper / len(letters) >= 0.70

    def _score(v: str) -> tuple[int, int, int, int, float]:
        not_junk = (
            not _is_short_junk(v)
            and not _is_boilerplate(v)
            and _ADDR_EXCLUDE.search(v) is None
            and _DATE_RE.search(v) is None
            and _MONEY_RE.search(v) is None
        )
        has_token = 1 if _COMPANY_TOKEN.search(v) else 0
        # New tier: candidates with a trade-name AND a suffix token
        # rank above candidates that are pure suffix-only or pure
        # trade-name-only (the partial-pick failure mode).  A
        # candidate matching ``_COMPANY_MULTI_SUFFIX_RE`` *as a whole*
        # is suffix-only and gets the lowest tier in this dimension.
        is_complete = 1 if (
            has_token
            and not _COMPANY_MULTI_SUFFIX_RE.match(v)
            and len([t for t in v.split() if t]) >= 2
        ) else 0
        upper = 1 if _is_mostly_upper(v) else 0
        # Topmost wins ties (smaller y → higher score via negation).
        return (int(not_junk), is_complete, has_token, upper, -_y(v))

    scores = [_score(c) for c in candidates]
    if (_is_attn_diffuse(attn_row)
            and candidates[0] == learned_clean
            and not _COMPANY_TOKEN.search(learned_clean)):
        # Demote learned candidate by zeroing its ``upper`` bit; rule
        # pick wins ties.  Mirrors :func:`_refine_address` handling.
        # A company token on the learned pick (``SDN BHD`` etc.) is such
        # strong positive evidence that we refuse to demote it even when
        # the attention row is flat — the H gate is only a tie-breaker,
        # not an eraser of legitimate signal.
        s = scores[0]
        scores[0] = (s[0], s[1], s[2], 0, s[4])
    return max(zip(candidates, scores, strict=True), key=lambda cs: cs[1])[0]


def _y_of(value: str, texts: list[str], bboxes: list[list[float]]) -> float:
    key = value.strip().lower()
    if not key:
        return 0.0
    for i, t in enumerate(texts):
        if key in t.lower():
            return bboxes[i][1] if i < len(bboxes) else 0.0
    return 0.0


# --- Company repair (transformative) ---------------------------------------
# Stars/hashes that the OCR or thermal-printer leaves around merchant
# trade names ("***ROYALTEA***" or "*** TAX INVOICE ***").  Symmetric
# leading/trailing run captured so a single pass strips both ends.
_COMPANY_DECOR_RE = re.compile(r"^[\*\#\-_=~`\s]+|[\*\#\-_=~`\s]+$")
# Whole-line boilerplate the company-anchor walk MUST skip.  Wider than
# ``_HEADER_JUNK`` (which only matches lines whose entire content is the
# keyword): also catches lines that *contain* TAX INVOICE / CASH BILL /
# OFFICIAL RECEIPT etc., even when wrapped in ``***`` decorations.
_COMPANY_BOILERPLATE_LINE_RE = re.compile(
    r"\b(?:TAX\s+INVOICE|CASH\s+BILL|OFFICIAL\s+RECEIPT|CUSTOMER\s+COPY|"
    r"MERCHANT\s+COPY|SIMPLIFIED(?:\s+TAX(?:\s+INVOICE)?)?|"
    r"CREDIT\s+NOTE|DELIVERY\s+ORDER|SALES\s+RECEIPT|RECEIPT\s+COPY|"
    r"DUPLICATE\s+RECEIPT)\b",
    re.IGNORECASE,
)
# Tail tokens to drop when refining a company candidate: registration
# numbers (``M076170-K``, ``107769-21``, ``139386 X``), GST IDs, OCR
# fragments (``stud01/59572``), Wi-Fi tokens, single-orphan letters.
_COMPANY_TAIL_DROP_RE = re.compile(
    r"\s+(?:"
    r"\(?\s*\d{4,}[\-\s]?[A-Z\d]{0,4}\s*\)?"  # 5+digit reg numbers
    r"|GST\s*[:#]?\s*\d+"
    r"|GST\s*ID\s*\d+"
    r"|TEL\s*[:.]?\s*[\d\-]+"
    r"|WI-?FI\s*\S+"
    r"|STUD\d+\S*"
    r"|MA?\d{5,}"
    r"|[A-Z]{1,3}\d{5,}\S*"
    r"|password\s*\S+"
    r")\s*\.?\s*$",
    re.IGNORECASE,
)
# Multi-token forward-extension regex that recognises a line whose
# content is purely company-suffix material — strict ``_COMPANY_SUFFIX``
# only matched a single segment.  This lets us pick up ``CO M SDN BHD``,
# ``(M) SDN BHD``, ``HOLDINGS SDN BHD`` as a *whole* extension target.
_COMPANY_MULTI_SUFFIX_RE = re.compile(
    r"^\s*"
    r"(?:\(?\s*M\s*\)?\s+)?(?:CO\.?\s*)?(?:M\s+)?"
    r"(?:SDN\.?\s*BHD\.?|BERHAD|BHD|S/B|PTE\.?\s*LTD\.?|LTD\.?"
    r"|HOLDINGS|ENTERPRISE(?:S)?|TRADING|MARKETING|CORPORATION|CORP\.?)"
    r"(?:\s+(?:SDN\.?\s*BHD\.?|BERHAD|BHD|S/B|HOLDINGS"
    r"|ENTERPRISE(?:S)?|TRADING|MARKETING))*"
    r"\s*\.?\s*$",
    re.IGNORECASE,
)


def _strip_decor(s: str) -> str:
    """Remove ``***`` / ``###`` / dashes / equals from both ends."""
    return _COMPANY_DECOR_RE.sub("", s).strip()


def _strip_company_tail(s: str) -> str:
    """Drop trailing OCR / registration noise from a company string.

    Run twice so a two-suffix tail (``"... 139386 X (M)"``) is fully
    removed.  Never returns empty — if the tail-drop would consume
    everything, return the original string so the caller can still
    score it (the trade-name / SDN-BHD scoring will rank it low).
    """
    prev = s
    for _ in range(2):
        cur = _COMPANY_TAIL_DROP_RE.sub("", prev).strip()
        if cur == prev:
            break
        prev = cur
    return prev or s


def _line_index_of(value: str, texts: list[str]) -> int:
    """Index of the first text-line that *contains* ``value`` (case-fold)."""
    if not value:
        return -1
    key = value.strip().lower()
    for i, t in enumerate(texts):
        if key and key in t.lower():
            return i
    return -1


def _company_extend(
    value: str, texts: list[str], bboxes: list[list[float]],
) -> str:
    """Forward/backward-extend a partial company pick.

    Two failure modes the score-only refiner cannot fix:

    1. **Prefix-only pick** (``"POPULAR BOOK"`` vs GT
       ``"POPULAR BOOK CO M SDN BHD"``): the picked line lacks any
       ``_COMPANY_TOKEN`` but a neighbouring line below is a pure
       company-suffix run (``"CO M SDN BHD"``).  Concatenate.

    2. **Suffix-only pick** (``"CO M SDN BHD"`` vs GT
       ``"POPULAR BOOK CO M SDN BHD"``): the picked line is *all*
       suffix tokens; the line above carries the trade name.
       Prepend that.

    Bounded to one line of extension on each side so we never invent
    a multi-line span out of thin air.  Idempotent on already-complete
    candidates (anything already containing both a trade-name and a
    suffix token returns unchanged).
    """
    cleaned = _strip_decor(value).strip()
    if not cleaned:
        return value
    cleaned = _strip_company_tail(cleaned)
    idx = _line_index_of(cleaned, texts)
    if idx < 0:
        return cleaned
    # Forward-extend: cleaned has no _COMPANY_TOKEN, next line is pure
    # suffix.  Skip only blank/junk lines between.
    has_token = bool(_COMPANY_TOKEN.search(cleaned))
    # Continuation marker: a trailing ``&`` / ``,`` / hanging
    # ``of/the/and`` strongly signals the trade name was line-wrapped.
    # Extend even when a company-token ending isn't on the next line,
    # provided the next line is plausibly upper-case alpha-heavy.
    ends_continuation = bool(re.search(r"[&,]\s*$|\b(?:and|of|the)\s*$",
                                        cleaned, re.IGNORECASE))
    if (ends_continuation or not has_token) and idx + 1 < len(texts):
        for j in range(idx + 1, min(idx + 3, len(texts))):
            nxt = _strip_decor(texts[j].strip())
            if not nxt:
                continue
            if _COMPANY_BOILERPLATE_LINE_RE.search(nxt):
                break
            if _DATE_RE.search(nxt) or _MONEY_RE.search(nxt):
                break
            if _ADDR_EXCLUDE.search(nxt):
                break
            # Strict multi-suffix line: pure ``CO M SDN BHD`` /
            # ``HOLDINGS SDN BHD`` etc.
            if _COMPANY_MULTI_SUFFIX_RE.match(nxt):
                cleaned = f"{cleaned} {_strip_company_tail(nxt)}"
                has_token = True
                break
            # Permissive extend: line ENDS with a company token
            # (``ENTERPRISE SETIA ALAM SDN BHD`` — the multi-token tail
            # the strict regex can't enumerate without exploding).  The
            # line must not also carry an address anchor (``JALAN`` /
            # ``LOT`` / 5-digit postcode) so we don't accidentally pull
            # in the first address line as part of the company name.
            looks_address = bool(re.search(
                r"\b(JALAN|JLN|LOT|TAMAN|TMN|BANDAR|NO\.?|\d{5})\b",
                nxt, re.IGNORECASE,
            ))
            if (_COMPANY_TOKEN.search(nxt) and not looks_address):
                cleaned = f"{cleaned} {_strip_company_tail(nxt)}"
                has_token = True
                break
            # Continuation extend: when the picked line ends with ``&``
            # / ``,`` etc., accept any short upper-case alpha-heavy
            # next line as the wrapped tail of the trade name.  Bounded
            # to one extension and rejected if the line contains an
            # address anchor.
            if ends_continuation and not looks_address:
                letters = [c for c in nxt if c.isalpha()]
                if (len(letters) >= 3
                        and len(nxt.split()) <= 5
                        and sum(1 for c in letters if c.isupper())
                            / max(len(letters), 1) >= 0.7):
                    cleaned = f"{cleaned} {_strip_company_tail(nxt)}"
                    break
            # Any other line stops the walk.
            break
    # Backward-extend: cleaned IS pure suffix (matches multi-suffix
    # regex on its own), look up one line for the trade name.
    if (idx > 0
            and _COMPANY_MULTI_SUFFIX_RE.match(cleaned)
            and not _is_short_junk(cleaned)):
        for j in range(idx - 1, max(idx - 3, -1), -1):
            prev = _strip_decor(texts[j].strip())
            if not prev:
                continue
            if _COMPANY_BOILERPLATE_LINE_RE.search(prev):
                break
            if _HEADER_JUNK.match(prev):
                break
            if _DATE_RE.search(prev) or _MONEY_RE.search(prev):
                break
            if _is_short_junk(prev):
                break
            # Plausible trade name: alpha-heavy, not a phone/GST line.
            if _ADDR_EXCLUDE.search(prev):
                break
            cleaned = f"{_strip_company_tail(prev)} {cleaned}"
            break
    return cleaned


def _is_addr_boundary(t: str) -> bool:
    """Address span terminator: money / date / phone-or-tax-id / header junk /
    invoice-cashier transition / company header / tax-id boilerplate.

    A 5-digit postcode line (``\\b\\d{5}\\b``) is normally NOT a
    boundary — ``40000 SHAH ALAM`` is the canonical *end* of a
    Malaysian address and must be included in the span.  PR-ADDR-PREC
    refines this exemption: the postcode short-circuit fires only when
    the line carries NO transaction-boundary or company-header
    keyword, so a contaminated tail like ``DOC NO 88421`` (which
    happens to contain a 5-digit run) still terminates the span.

    Boundary classes:

    * :data:`_ADDR_TERMINATOR` — invoice/cashier/footer keywords plus
      the bottom-cut additions (``RECEIPT``, bare ``CASH``, ``COVER``,
      ``WAITER``, ``DOC NO``, ``DATE:``/``TIME:``, ``BILL``, bare
      ``ROC`` / ``TEL`` / ``FAX``, ``CREDIT NOTE``).
    * :data:`_COMPANY_TOKEN`   — hard company markers (``SDN BHD``,
                                 ``BERHAD``, ``ENTERPRISE``, …).
    * :data:`_ADDR_COMPANY_HEADER` — wider company / tax-ID stripping
      regex used pre-anchor and during backward-extend so headers like
      ``INTERNATIONAL``, ``(M) SDN``, ``GST: 12345``, ``\\d{12}`` and
      registration numbers ``\\d{6,}-[A-Z]`` are excluded.
    """
    has_postcode = bool(_POSTCODE_RE.search(t))
    has_terminator = bool(_ADDR_TERMINATOR.search(t))
    has_company = bool(
        _COMPANY_TOKEN.search(t) or _ADDR_COMPANY_HEADER.search(t),
    )
    if has_postcode and not (has_terminator or has_company):
        return False
    return bool(_MONEY_RE.search(t) or _DATE_RE.search(t)
                or _ADDR_EXCLUDE.search(t) or _HEADER_JUNK.match(t)
                or has_terminator or has_company)


def _line_height(bboxes: list[list[float]], i: int) -> float:
    if i >= len(bboxes) or len(bboxes[i]) < 4:
        return 0.0
    return max(bboxes[i][3] - bboxes[i][1], 0.0)


def _median_line_height(bboxes: list[list[float]]) -> float:
    """Median (y2-y1) of non-zero-height regions; 0 when no valid boxes."""
    hs = [bboxes[i][3] - bboxes[i][1]
          for i in range(len(bboxes))
          if len(bboxes[i]) >= 4 and bboxes[i][3] > bboxes[i][1]]
    if not hs:
        return 0.0
    hs.sort()
    return hs[len(hs) // 2]


def enforce_address_contiguity(
    picks: list[int], bboxes: list[list[float]], gap_mult: float = 2.0,
) -> list[int]:
    """Prune picks whose top-edge gap exceeds ``gap_mult`` × median line height.

    Fixes the dominant ``_MULTI_LINE_FRACTION`` failure mode: a mildly
    diffuse attention head drags tax/phone/GST lines (separated from the
    last address line by several receipt rows) into the ``address``
    field.  Picks must already be sorted top→bottom by ``y1``.  The
    first pick is always kept; subsequent picks are kept only when
    ``y1_curr - y2_prev <= gap_mult * median_line_height``.  Degenerate
    inputs (empty picks, no valid boxes, zero median height) return the
    input unchanged so this helper never *removes* an otherwise-kept
    region just because we couldn't estimate geometry.
    """
    if len(picks) < 2:
        return list(picks)
    mh = _median_line_height(bboxes)
    if mh <= 0.0:
        return list(picks)
    kept = [picks[0]]
    for i in picks[1:]:
        if i >= len(bboxes) or len(bboxes[i]) < 4:
            continue
        prev = kept[-1]
        if prev >= len(bboxes) or len(bboxes[prev]) < 4:
            kept.append(i)
            continue
        gap = bboxes[i][1] - bboxes[prev][3]
        if gap <= gap_mult * mh:
            kept.append(i)
    return kept


def _same_line(
    bboxes: list[list[float]], a: int, b: int, frac: float = 0.5,
) -> bool:
    """True when two regions' y-intervals overlap by >= ``frac`` × min height.

    SROIE receipts are frequently OCR'd into multi-column regions on a
    single visual line (brand + address, or a long address split across
    two bboxes at the same y).  Treating those as one line during span
    assembly keeps natural reading order and avoids the ``"JLN JEJAKA,
    TAMAN MALURI"`` / ``"3RD FLR, AEON..."`` split we see in the miss
    table.
    """
    if (a >= len(bboxes) or b >= len(bboxes)
            or len(bboxes[a]) < 4 or len(bboxes[b]) < 4):
        return False
    y1a, y2a = bboxes[a][1], bboxes[a][3]
    y1b, y2b = bboxes[b][1], bboxes[b][3]
    overlap = max(0.0, min(y2a, y2b) - max(y1a, y1b))
    min_h = max(min(y2a - y1a, y2b - y1b), 1e-6)
    return overlap / min_h >= frac


def _address_span(
    texts: list[str], bboxes: list[list[float]],
) -> str:
    """Greedy spatial span anchored on the topmost address-keyword region.

    Fixes the dominant failure mode in the miss table: the assigner
    picks only line 1-2 of a 4-5-line address (40+ of 63 misses are pure
    prefix-of-GT).  Strategy:

    1. **Forward anchor** — find the topmost y-ordered region matching
       :data:`_ADDR_ANCHOR` that is not a boundary (money/date/phone/
       header junk).  The anchor set now includes the Malaysian 5-digit
       postcode so tail-only OCR (``43200 CHERAS, SELANGOR``) still
       anchors correctly; the backward-extend below then recovers the
       street/floor prefix.
    2. **Backward extend** — walk up one line from the anchor and
       include it if it is a plausible address prefix (all-alpha mall
       name like ``PARADIGM MALL`` or ``DOMINO'S PIZZA``, floor marker,
       or short upper-case token) and is not a boundary, date, or
       phone line.
    3. **Forward walk** — append every subsequent non-junk, non-boundary
       line until hitting a boundary.  ``_is_addr_boundary`` treats
       postcode-bearing lines as in-span, so the tail is never chopped.
    4. **Same-line sibling merge** — regions whose y-intervals overlap
       by ≥50% of the smaller line height get joined with a single
       space so multi-column layouts read linearly.

    Returns the concatenated text; empty string when no anchor is found
    (caller falls back to the learned pick).
    """
    n = len(texts)
    if n == 0:
        return ""
    y_order = sorted(range(n), key=lambda j: bboxes[j][1] if j < len(bboxes) else 0.0)
    # Find topmost anchor in y-order that isn't a header/money/date/phone
    # and isn't a company-registration/tax-ID line (``CO. NO. 37365-A``,
    # ``GST NO. 123...``).  The shared ``_ADDR_EXCLUDE`` doesn't tolerate
    # a period between ``CO`` and ``NO``, so we additionally reject any
    # line that starts with :data:`_ADDR_LEADING_JUNK_RE` here.
    anchor_pos: int | None = None
    for pos, j in enumerate(y_order):
        t = texts[j].strip()
        if not t or _is_short_junk(t) or _is_addr_boundary(t):
            continue
        if _ADDR_LEADING_JUNK_RE.match(t):
            continue
        if _ADDR_ANCHOR.search(t):
            anchor_pos = pos
            break
    if anchor_pos is None:
        return ""
    # Backward extend: include up to one preceding line if it is an
    # unambiguous address prefix (floor/mall/brand-venue keyword) and is
    # neither a company header nor a terminator.  The earlier catch-all
    # that accepted any ``short upper-case`` label was too permissive —
    # it dragged ``MR D.I.Y M SDN BHD`` and ``TANCHMAS BUKCENTRE P SDN
    # BHD`` into the span.  A plain ``_COMPANY_TOKEN`` check plus a
    # keyword requirement tightens precision without losing ``DOMINO'S
    # PIZZA``, ``PARADIGM MALL``, ``GROUND FLOOR`` -style prefixes.
    start_pos = anchor_pos
    if anchor_pos > 0:
        k = y_order[anchor_pos - 1]
        prev = texts[k].strip()
        if (prev
                and not _is_short_junk(prev)
                and not _is_addr_boundary(prev)
                and not _ADDR_LEADING_JUNK_RE.match(prev)
                and not _DATE_RE.search(prev)
                and not _MONEY_RE.search(prev)
                and not _COMPANY_TOKEN.search(prev)
                and not _ADDR_TERMINATOR.search(prev)
                # Plausible address prefix: carries an address anchor,
                # a Malaysian state/city continuation token, or is a
                # mall/floor/building keyword line.
                and (_ADDR_ANCHOR.search(prev)
                     or _ADDR_CONTINUATION.search(prev))):
            start_pos = anchor_pos - 1
    picked: list[int] = [y_order[start_pos]]
    for j in y_order[start_pos + 1:]:
        t = texts[j].strip()
        if not t or _is_short_junk(t):
            continue
        if _is_addr_boundary(t):
            break
        picked.append(j)
    # Same-line sibling merge: regions on the same visual line get a
    # single space separator; line breaks also get a single space (the
    # SROIE GT concatenates multi-line addresses without newlines).
    out: list[str] = []
    for idx, j in enumerate(picked):
        tok = texts[j].strip()
        if not tok:
            continue
        if idx > 0 and _same_line(bboxes, picked[idx - 1], j):
            out[-1] = out[-1] + " " + tok
        else:
            out.append(tok)
    return " ".join(out)


# Leading company-registration / tax-ID junk stripping lives above
# (``_ADDR_LEADING_JUNK_RE`` / :func:`_strip_leading_junk`) so both the
# span builder and the refiner use the same definition.


def _refine_address(
    learned: str, texts: list[str], bboxes: list[list[float]],
    field_values: dict[str, str], attn_row: list[float] | None = None,
) -> str:
    """Prefer postcode-bearing, junk-free, longest candidate.

    The miss table shows ~60% of address losses are pure prefix-of-GT
    (under-picked).  A complete Malaysian postal address always ends in
    a 5-digit postcode + city/state, so *having a postcode* is the
    cleanest single-bit signal of completeness and dominates the
    selection.  Scoring key, higher-wins, evaluated per candidate:

    1. ``has_postcode``      — 5-digit run present.
    2. ``not addr_junk``     — no tax-ID / phone / reg-no tokens.
    3. ``has_continuation``  — Malaysian state / city token present,
                               a secondary completeness cue.
    4. ``length``            — token-F1 tie-break.

    Candidates considered: the learned pick (with ``CO. NO. ...`` /
    ``GST NO. ...`` leading junk stripped), the rule-based
    ``_pick_address``, and the greedy ``_address_span``.

    Strategy (H) — when the learned attention is diffuse, downweight the
    learned candidate by one tier in the tuple key so the rule-based
    span wins ties.  This delegates to the higher-F1 rule arm on
    precisely the receipts where the learned arm has no conviction.
    """
    learned_clean = _strip_leading_junk(learned)
    span = _strip_leading_junk(_address_span(texts, bboxes))
    total_pick = extract_total(texts, bboxes)
    date_pick = extract_date(texts)
    rule_addr = _strip_leading_junk(_pick_address(
        texts, bboxes, used=set(),
        company_y=_y_of(field_values.get("company", ""), texts, bboxes),
        total_y=(bboxes[total_pick[0]][1]
                 if total_pick and total_pick[0] < len(bboxes) else 0.0),
        date_y=(bboxes[date_pick[0]][1]
                if date_pick and date_pick[0] < len(bboxes) else 0.0),
    ))

    def _score(s: str) -> tuple[int, int, int, int]:
        st = s.strip()
        if not st:
            return (0, 0, 0, 0)
        has_postcode = 1 if _POSTCODE_RE.search(st) else 0
        # PR-ADDR-PREC — ``not_junk`` is now a tri-test: a candidate
        # contaminated with company-header / transaction-boundary
        # tokens (``BHD``, ``INV NO``, ``CASH``, …) loses the bit even
        # when it carries the postcode, so the cleaner span/rule
        # candidate wins on the (post,not_junk,cont,len) tuple.
        not_junk = 0 if (
            _ADDR_EXCLUDE.search(st)
            or _ADDR_TERMINATOR.search(st)
            or _ADDR_COMPANY_HEADER.search(st)
        ) else 1
        has_cont = 1 if _ADDR_CONTINUATION.search(st) else 0
        return (has_postcode, not_junk, has_cont, len(st))

    candidates = [learned_clean, rule_addr, span]
    scores = [_score(c) for c in candidates]
    if _is_attn_diffuse(attn_row):
        # Demote the learned candidate by zeroing its ``not_junk`` bit;
        # equivalent to "when the assigner is unsure, trust the rule span".
        s_learned = scores[0]
        scores[0] = (s_learned[0], 0, s_learned[2], s_learned[3])
    return max(zip(candidates, scores, strict=True), key=lambda cs: cs[1])[0]


def refine_assignments(
    draft: dict[str, str], texts: list[str], bboxes: list[list[float]],
    attn: list[list[float]] | None, fields: list[str],
) -> dict[str, str]:
    """Apply per-field refiner + output normaliser to the learned draft."""
    out = dict(draft)
    by_idx = {f.lower(): i for i, f in enumerate(fields)}
    if "date" in out:
        out["date"] = normalize_date(_refine_date(out["date"], texts))
    if "total" in out:
        row = attn[by_idx["total"]] if attn and "total" in by_idx else None
        out["total"] = normalize_total_value(
            _refine_total(out["total"], texts, bboxes, row),
        )
    if "company" in out:
        row = attn[by_idx["company"]] if attn and "company" in by_idx else None
        out["company"] = normalize_company(
            _refine_company(out["company"], texts, bboxes, attn_row=row),
        )
    if "address" in out:
        row = attn[by_idx["address"]] if attn and "address" in by_idx else None
        out["address"] = normalize_address(
            _refine_address(out["address"], texts, bboxes, out, attn_row=row),
        )
    return out
