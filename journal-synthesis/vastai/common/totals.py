"""Prior-work-faithful gold/predicted total extraction + correctness.

THE CORRECTNESS BUG this fixes: every experiment derived a receipt's
"correct" flag from `decoded_fields == gold_annotation` (two structurally
different dicts -- the Donut `token2json` nested {menu,total,sub_total}
envelope vs. the flat canonical {"fields": {...}} annotation), so
`n_correct == 0` for EVERY policy/verifier/corpus, and the gold total
was read from `gt.get("gt_parse", gt).get("total")` (a key that does not
exist in the on-disk canonical annotation), so `gold_total` was always
None and the subset-sum axis abstained vacuously.

This module is the SINGLE source of truth for:

  * `parse_money`        -- string -> integer cents
  * `flatten_donut`      -- Donut token2json envelope -> flat fields
  * `gold_fields`        -- canonical annotation -> flat fields
  * `gold_total_cents`   -- gold total in cents (or None)
  * `pred_total_cents`   -- predicted total in cents (or None)
  * `items_cents`        -- item prices in cents
  * `is_correct`         -- gold/pred total equal within EPS_CENTS
  * `subset_sum_exists`  -- Paper-1 subset-sum verifier

Everything here is lifted, byte-faithful in METHOD and CONVENTION, from
the prior papers (read-only repos, not modified):

  parse_money / parse_items / subset_sum_exists / EPS_CENTS == 1
      arith-gating/attic/experiments/phase4_verify_arith.py
      (the I1-I5 verifier, itself "derived from the I1-I5 verifier in
       triology/paper3").

  _flatten_gt  (CORD Donut `gt_parse` / `menu` / `sub_total` / `total`
                -> {total, subtotal, tax, paid, change, items})
      arith-gating/experiments/phase3_donut_extract.py::_flatten_gt
      (identical logic to scripts/fetch_data.py::_decode_donut_gt that
       WROTE the on-disk gold annotations).

  WildReceipt canonical field map (`Total_value` -> `total`, etc.)
      arith-gating/scripts/fetch_wildreceipt.py::to_canonical, which
      writes the SAME flat {"id","image_filename","tokens","fields":{
      total,subtotal,tax,items}} schema as the CORD fetcher, so a SINGLE
      gold extractor serves both corpora.

NOTHING here is loosened to favour the paper's thesis. The tolerance is
EPS_CENTS == 1 exactly as the prior work uses it; "correct" is an exact
cents identity within that single-cent tolerance. If the honest result
is precision 0, that is the result.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Tolerance, byte-faithful to arith-gating/attic/.../phase4_verify_arith.py
# (EPS_CENTS = 1) which itself mirrors triology / Paper 1. NOT widened.
EPS_CENTS = 1

# CORD-v2 (Indonesian restaurant receipts) admissible PPN tax rates,
# byte-faithful to phase4_verify_arith.ADMISSIBLE_RATES["cord"].
ADMISSIBLE_RATES_CORD = (0.0, 0.10, 0.11)


# --------------------------------------------------------------------------
# parse_money / parse_items -- VERBATIM convention from
# arith-gating/attic/experiments/phase4_verify_arith.py
# --------------------------------------------------------------------------

def parse_money(s: Any) -> Optional[int]:
    """Parse a currency string to integer cents. Returns None on failure.

    VERBATIM from phase4_verify_arith.parse_money: a fractional part is
    recognised ONLY when it is 1-2 digits (`[.,]\\d{1,2}`). For CORD
    Indonesian Rupiah strings like '60.000' the regex therefore matches
    just '60' (the '.000' is NOT a 2-digit fraction), yielding 6000
    cents -- and the SAME parser is applied to BOTH gold and prediction
    so the comparison is internally consistent exactly as in the prior
    work. We deliberately do NOT "fix" this to a different cents
    convention: matching the prior papers' definition of correct is the
    requirement; changing it would silently change every reported
    number.
    """
    if s is None:
        return None
    m = re.search(r"(-?\d+(?:[.,]\d{1,2})?)", str(s).replace(" ", ""))
    if not m:
        return None
    val = m.group(1).replace(",", ".")
    try:
        return int(round(float(val) * 100))
    except ValueError:
        return None


def parse_items(s: Any) -> List[int]:
    """Pull a list of cent-amounts out of a free-form items string.
    VERBATIM from phase4_verify_arith.parse_items."""
    if s is None:
        return []
    out: List[int] = []
    for tok in re.findall(r"-?\d+(?:[.,]\d{1,2})?", str(s)):
        v = parse_money(tok)
        if v is not None:
            out.append(v)
    return out


# --------------------------------------------------------------------------
# _flatten_gt -- VERBATIM from
# arith-gating/experiments/phase3_donut_extract.py::_flatten_gt
# (identical to scripts/fetch_data.py::_decode_donut_gt which WROTE the
#  on-disk CORD gold annotations).
# --------------------------------------------------------------------------

def flatten_donut(gt: Any) -> Dict[str, str]:
    """Donut CORD token2json envelope -> flat {total,subtotal,tax,paid,
    change,items}. VERBATIM _flatten_gt logic."""
    if isinstance(gt, dict) and "gt_parse" in gt:
        gt = gt["gt_parse"]
    if not isinstance(gt, dict):
        return {}
    out: Dict[str, str] = {}
    if isinstance(gt.get("sub_total"), dict):
        st = gt["sub_total"]
        if "subtotal_price" in st:
            out["subtotal"] = str(st["subtotal_price"])
        if "tax_price" in st:
            out["tax"] = str(st["tax_price"])
    if isinstance(gt.get("total"), dict):
        t = gt["total"]
        if "total_price" in t:
            out["total"] = str(t["total_price"])
        if "cashprice" in t:
            out["paid"] = str(t["cashprice"])
        if "changeprice" in t:
            out["change"] = str(t["changeprice"])
    if isinstance(gt.get("menu"), list):
        prices = [str(it.get("price", "")) for it in gt["menu"]
                  if isinstance(it, dict) and "price" in it]
        if prices:
            out["items"] = " ".join(prices)
    return out


def gold_fields(gold: Any) -> Dict[str, str]:
    """Flat {total,subtotal,tax,...} from whatever the decode-once cache
    stored as the raw gold annotation.

    The on-disk CORD/WildReceipt annotation that the loaders read is the
    canonical schema produced by arith-gating fetch_data /
    fetch_wildreceipt: {"id","image_filename","tokens","fields":{...}}.
    The flat field dict is therefore `gold["fields"]`. We also accept a
    raw Donut `gt_parse` envelope (older caches / robustness) and a
    bare flat dict, so a single extractor is correct for both corpora
    regardless of which producer wrote the annotation.
    """
    if not isinstance(gold, dict):
        return {}
    f = gold.get("fields")
    if isinstance(f, dict) and f:
        return {k: ("" if v is None else str(v)) for k, v in f.items()}
    # Donut gt_parse envelope (menu/sub_total/total) -> flatten it.
    if "gt_parse" in gold or "menu" in gold or "sub_total" in gold:
        flat = flatten_donut(gold)
        if flat:
            return flat
    # Already-flat fallback (e.g. {"total": "...", ...}).
    if any(k in gold for k in
           ("total", "subtotal", "tax", "paid", "change", "items")):
        return {k: ("" if v is None else str(v)) for k, v in gold.items()
                if not isinstance(v, (dict, list))}
    return {}


# --------------------------------------------------------------------------
# Total / items extraction + correctness
# --------------------------------------------------------------------------

def gold_total_cents(gold: Any) -> Optional[int]:
    """Gold receipt total in cents, prior-work definition."""
    return parse_money(gold_fields(gold).get("total"))


def gold_items_cents(gold: Any) -> List[int]:
    """Gold item prices in cents (joined string in canonical 'items')."""
    return parse_items(gold_fields(gold).get("items"))


def gold_tax_cents(gold: Any) -> int:
    """Gold tax in cents, 0 if absent (subset-sum tau)."""
    return parse_money(gold_fields(gold).get("tax")) or 0


def pred_total_cents(pred_fields: Any) -> Optional[int]:
    """Predicted receipt total in cents from the Donut token2json
    envelope, prior-work definition (flatten then parse_money)."""
    return parse_money(flatten_donut(pred_fields).get("total"))


def pred_items_cents(pred_fields: Any) -> List[int]:
    """Predicted item prices in cents from the Donut token2json
    envelope (menu[].price), prior-work definition."""
    return parse_items(flatten_donut(pred_fields).get("items"))


def pred_tax_cents(pred_fields: Any) -> int:
    """Predicted tax in cents, 0 if absent (subset-sum tau)."""
    return parse_money(flatten_donut(pred_fields).get("tax")) or 0


def is_correct(gold: Any, pred_fields: Any,
               eps_cents: int = EPS_CENTS) -> bool:
    """A receipt is CORRECT iff the predicted total equals the gold total
    within EPS_CENTS (the prior papers' single-cent tolerance). This is
    the prior-work definition of an end-to-end-correct receipt total; it
    is NOT the structural `decoded_dict == gold_dict` comparison (which
    was the bug). If gold or pred total is unparseable the receipt is
    NOT correct (a missing/garbled total is not a correct extraction).
    """
    g = gold_total_cents(gold)
    p = pred_total_cents(pred_fields)
    if g is None or p is None:
        return False
    return abs(g - p) <= eps_cents


# --------------------------------------------------------------------------
# subset_sum_exists -- VERBATIM from
# arith-gating/attic/experiments/phase4_verify_arith.py::subset_sum_exists
# --------------------------------------------------------------------------

def subset_sum_exists(items: List[int], target: int,
                      tol: int = EPS_CENTS) -> bool:
    """Pseudo-polynomial subset-sum: True iff some subset of `items` sums
    to within `tol` of `target`. VERBATIM from phase4_verify_arith. Note
    the prior work's convention: with no items the predicate is
    `target <= tol` (NOT a vacuous abstain) -- this is what stops the
    Axis-A verifier abstaining on everything."""
    if target < 0:
        return False
    if not items:
        return target <= tol
    reachable = {0}
    for v in items:
        reachable |= {r + v for r in reachable}
    return any(abs(r - target) <= tol for r in reachable)


def subset_sum_verdict_prior(candidate_cents: Optional[int],
                             items_cents: Optional[List[int]],
                             tau_cents: int = 0) -> str:
    """Axis-A verdict using the PRIOR-WORK subset_sum_exists semantics.

    Returns "pass" | "fail" | "abstain". We abstain ONLY when there is no
    candidate total to test (genuinely unavailable identity); otherwise
    we apply phase4_verify_arith's predicate against the subtotal-style
    target `candidate - tau` (so a stated tax is accounted for, matching
    the prior I3 which sums items to the *subtotal*). This deliberately
    does NOT add the extra `len(items) < 2` / k_min abstain gate that the
    old common.pipeline.subset_sum_verdict used to impose -- that gate
    (plus the always-None pred/gold totals) is exactly why E5 Axis-A had
    n_accept == 0. We use the prior paper's verifier as-is; we do not
    invent a stricter or looser rule.
    """
    if candidate_cents is None:
        return "abstain"
    items = items_cents or []
    target = candidate_cents - (tau_cents or 0)
    return "pass" if subset_sum_exists(items, target) else "fail"
