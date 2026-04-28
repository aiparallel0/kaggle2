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
_ATTN_DIFFUSE_MARGIN = 0.05
_TOTAL_OVERRIDE_MARGIN_DIFFUSE = 0.5


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


def _score_money(
    i: int, texts: list[str], rank: dict[int, int], money_idxs: list[int],
    attn_row: list[float] | None = None,
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
    if _SUBTOTAL_KW_RE.search(nbr):
        s -= 4.0
    if _TOTAL_NEGATIVE.search(nbr):
        s -= 2.0
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
    return s


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
    * the best-scoring candidate beats the learned one by at least the
      override margin — which drops from :data:`_TOTAL_OVERRIDE_MARGIN`
      to :data:`_TOTAL_OVERRIDE_MARGIN_DIFFUSE` when the assigner's
      attention is diffuse (strategy H — confidence-gated delegation).

    The margin-based rule naturally keeps the learned pick on receipts
    where the scorer finds no decisive signal, while still correcting
    the classic SUBTOTAL-vs-GRAND-TOTAL confusion when ``_TOTAL_STRONG``
    matches only the right line.
    """
    repaired = [repair_money_ocr(t) for t in texts]
    money_idxs = [i for i, t in enumerate(repaired) if _MONEY_RE.search(t)]
    if not money_idxs:
        return learned
    rank = _attn_rank(attn_row) if attn_row else {}
    scored: list[tuple[float, int, str]] = sorted(
        ((_score_money(i, repaired, rank, money_idxs, attn_row), i,
          _MONEY_RE.search(repaired[i]).group(0))  # type: ignore[union-attr]
         for i in money_idxs),
        reverse=True,
    )
    best_score, _, best_val = scored[0]
    learned_clean = _strip_currency(learned)
    if not _MONEY_RE.fullmatch(learned_clean):
        # Learned value isn't a usable money string; take the scored
        # pick unconditionally (fall back to learned if no positive
        # evidence either).
        return _strip_currency(best_val) if best_score > 0 else learned_clean
    # Learned value is a well-formed money string — protect it unless a
    # competing candidate is decisively better.  The required margin is
    # relaxed when the assigner is unsure of itself (flat attention row).
    learned_score = next(
        (sc for sc, _, v in scored if v.strip() == learned_clean), float("-inf"),
    )
    margin = (_TOTAL_OVERRIDE_MARGIN_DIFFUSE if _is_attn_diffuse(attn_row)
              else _TOTAL_OVERRIDE_MARGIN)
    if best_score - learned_score >= margin and best_score > 0:
        return _strip_currency(best_val)
    return learned_clean


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
    learned_clean = learned.strip()
    picked = _pick_company(texts, bboxes, used=set())
    rule_pick = picked[1] if picked is not None else ""
    candidates: list[str] = []
    seen: set[str] = set()
    for c in (learned_clean, rule_pick):
        k = c.lower()
        if c and k not in seen:
            seen.add(k)
            candidates.append(c)
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

    def _score(v: str) -> tuple[int, int, int, float]:
        not_junk = (
            not _is_short_junk(v)
            and _HEADER_JUNK.match(v) is None
            and _ADDR_EXCLUDE.search(v) is None
            and _DATE_RE.search(v) is None
            and _MONEY_RE.search(v) is None
        )
        has_token = 1 if _COMPANY_TOKEN.search(v) else 0
        upper = 1 if _is_mostly_upper(v) else 0
        # Topmost wins ties (smaller y → higher score via negation).
        return (int(not_junk), has_token, upper, -_y(v))

    scores = [_score(c) for c in candidates]
    if (_is_attn_diffuse(attn_row)
            and candidates[0] == learned_clean
            and not _COMPANY_TOKEN.search(learned_clean)):
        # Demote learned candidate by zeroing its ``is_mostly_upper`` bit;
        # rule pick wins ties.  Mirrors :func:`_refine_address` handling.
        # A company token on the learned pick (``SDN BHD`` etc.) is such
        # strong positive evidence that we refuse to demote it even when
        # the attention row is flat — the H gate is only a tie-breaker,
        # not an eraser of legitimate signal.
        s = scores[0]
        scores[0] = (s[0], s[1], 0, s[3])
    return max(zip(candidates, scores, strict=True), key=lambda cs: cs[1])[0]


def _y_of(value: str, texts: list[str], bboxes: list[list[float]]) -> float:
    key = value.strip().lower()
    if not key:
        return 0.0
    for i, t in enumerate(texts):
        if key in t.lower():
            return bboxes[i][1] if i < len(bboxes) else 0.0
    return 0.0


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
