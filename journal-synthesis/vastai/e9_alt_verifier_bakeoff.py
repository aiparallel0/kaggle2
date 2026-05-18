#!/usr/bin/env python3
"""E9 - Alternative structural-verifier bake-off vs subset-sum.

WHAT IT COMPUTES
  Tests PREREGISTRATION.md H4: subset-sum (I3) is not dominated by an
  alternative structural verifier on precision at matched coverage.
  Implements three alternatives, faithful to the triology identities
  module (lifted method):
    line_item_qty : sum(price_i * qty_i) ~= total            (I3-like)
    subtotal_tax  : subtotal + tax ~= total                  (I2 / I4)
    rounding      : total is a valid rounding of subtotal+tax (cash-
                    rounding rule, +/- the corpus rounding increment)
  All four verifiers are run on the SAME decoded receipts (one pipeline,
  receipt_ids aligned by construction). Reports per-verifier precision at
  matched coverage with Wilson CIs, and pairwise verdict orthogonality
  (phi/MCC + two-sided permutation p) vs subset-sum.

WHAT IT NEEDS
  GPU. KIE checkpoint (--checkpoint). A corpus whose annotations carry
  per-line-item price+qty (CORD-v2 menu list does; that is exactly the
  structured data E9 was BLOCKED for lacking on the stored arith-gating
  fields - here it is decoded fresh, so it is available).

EXPECTED OUTPUTS
  vastai/results/E9_records.jsonl - unified per-receipt records (+ each
                                    verifier verdict in extra)
  vastai/results/E9.json          - per-verifier precision / coverage /
                                    orthogonality vs subset-sum

STATUS: NOT RUN. Needs GPU inference to recover per-line-item structure.
Computes-and-writes; never hardcodes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (  # noqa: E402
    UnifiedRecord, write_records, write_result, seed_everything,
    phi_mcc, perm_p, wilson,
    decode_or_load, flatten_donut, parse_money,
    gold_total_cents, pred_total_cents, pred_items_cents, pred_tax_cents,
    is_correct, subset_sum_verdict_prior, EPS_CENTS,
)

# EPS_CENTS is the prior-work single-cent tolerance (common.totals,
# phase4_verify_arith). Was locally 2 here; tightening to the prior
# work's 1 is NOT a thesis-favouring loosening (it can only make the
# alt-verifiers stricter).


def v_line_item_qty(fields):
    """sum(price_i * qty_i) within EPS of total. Returns pass|fail|abstain
    (abstain when no parseable line items / total).

    `fields` is the raw Donut token2json envelope; the menu list lives
    at the top level, but the receipt total is nested under
    total.total_price -- so we read the total via the prior-work
    flatten+parse_money, not a flat `fields["total"]` (which was None
    on the nested envelope)."""
    flat = flatten_donut(fields)
    total = parse_money(flat.get("total"))
    menu = fields.get("menu") or fields.get("items_detail")
    if total is None or not isinstance(menu, list) or not menu:
        return "abstain"
    acc = 0
    seen = False
    for it in menu:
        if not isinstance(it, dict):
            continue
        price = parse_money(it.get("price"))
        if price is None:
            continue
        qty_raw = it.get("cnt") or it.get("qty") or 1
        try:
            qty = int(float(str(qty_raw).split()[0]))
        except (ValueError, IndexError):
            qty = 1
        acc += price * qty
        seen = True
    if not seen:
        return "abstain"
    return "pass" if abs(acc - total) <= EPS_CENTS else "fail"


def v_subtotal_tax(fields):
    """subtotal + tax within EPS of total (triology I2/I4 method).
    Reads the nested Donut envelope via the prior-work flatten."""
    flat = flatten_donut(fields)
    total = parse_money(flat.get("total"))
    sub = parse_money(flat.get("subtotal"))
    tax = parse_money(flat.get("tax")) or 0
    if total is None or sub is None:
        return "abstain"
    return "pass" if abs(sub + tax - total) <= EPS_CENTS else "fail"


def v_rounding(fields, increment_cents=5):
    """total is a valid cash-rounding of subtotal+tax to the nearest
    `increment_cents` (default 5c). abstain if inputs missing."""
    flat = flatten_donut(fields)
    total = parse_money(flat.get("total"))
    sub = parse_money(flat.get("subtotal"))
    tax = parse_money(flat.get("tax")) or 0
    if total is None or sub is None:
        return "abstain"
    raw = sub + tax
    rounded = int(round(raw / increment_cents) * increment_cents)
    return "pass" if abs(rounded - total) <= EPS_CENTS else "fail"


def parse_args():
    ap = argparse.ArgumentParser(description="E9 alt-verifier bake-off")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--corpus", required=True, help="label=path")
    ap.add_argument("--rounding_increment_cents", type=int, default=5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_records",
                    default=os.path.join(HERE, "results", "E9_records.jsonl"))
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "E9.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    # Decode-once shared cache (identical decode; loaded if already
    # produced, NO model/GPU on a cache hit). Verifier math below is
    # byte-for-byte the original, just sourced from the shared primitives.
    primitives = decode_or_load(
        args.corpus, args.checkpoint, args.task_prompt, args.batch)
    label, _path = args.corpus.split("=", 1)
    backbone = os.path.basename(args.checkpoint.rstrip("/"))

    records = []
    for p in primitives:
        rid = p["receipt_id"]
        fields = p["fields"] if isinstance(p["fields"], dict) else {}
        sm, cs = p["softmax_confidence"], p["c_seq"]
        gt = p["gold"]
        pred_total = pred_total_cents(fields)
        items = pred_items_cents(fields)
        tau = pred_tax_cents(fields)
        ss = subset_sum_verdict_prior(pred_total, items, tau)
        records.append(UnifiedRecord(
            receipt_id=f"{label}:{rid}", corpus=label,
            backbone=backbone,
            gold_total=gold_total_cents(gt),
            pred_total=pred_total, softmax_confidence=sm, c_seq=cs,
            arith_pass=(ss == "pass"), subset_sum_verdict=ss,
            beam_margin=None,
            extra={
                "correct": is_correct(gt, fields),
                "v_line_item_qty": v_line_item_qty(fields),
                "v_subtotal_tax": v_subtotal_tax(fields),
                "v_rounding": v_rounding(
                    fields, args.rounding_increment_cents)}))

    write_records(args.out_records, records)

    verifiers = ["subset_sum", "v_line_item_qty", "v_subtotal_tax",
                 "v_rounding"]

    def verdict(r, v):
        return (r.subset_sum_verdict if v == "subset_sum"
                else r.extra[v])

    summary = {}
    for v in verifiers:
        appl = [r for r in records if verdict(r, v) != "abstain"]
        acc = [r for r in appl if verdict(r, v) == "pass"]
        corr = sum(1 for r in acc if r.extra.get("correct"))
        n = len(acc)
        lo, hi = wilson(corr, n) if n else (None, None)
        summary[v] = {
            "n_applicable": len(appl), "n_accept": n, "n_correct": corr,
            "precision": (corr / n) if n else None,
            "coverage": (len(appl) / len(records)) if records else None,
            "precision_wilson95": [lo, hi]}

    # orthogonality vs subset-sum on jointly-applicable receipts
    ortho = {}
    for v in verifiers:
        if v == "subset_sum":
            continue
        joint = [r for r in records
                 if r.subset_sum_verdict != "abstain"
                 and r.extra[v] != "abstain"]
        a = [0 if r.subset_sum_verdict == "pass" else 1 for r in joint]
        b = [0 if r.extra[v] == "pass" else 1 for r in joint]
        phi, counts = phi_mcc(a, b)
        ortho[f"subset_sum_vs_{v}"] = {
            "n_joint": len(joint), "phi_mcc": phi,
            "counts_n11_n10_n01_n00": counts,
            "permutation_p_two_sided": perm_p(a, b, seed=args.seed)}

    payload = {
        "experiment": "E9",
        "scope": (f"alternative structural-verifier bake-off vs subset-sum "
                  f"on shared decoded receipts (H4); corpus={label}; "
                  f"path={_path}; n_records={len(records)} "
                  f"(split/size is whatever --corpus points at)"),
        "per_verifier": summary,
        "orthogonality_vs_subset_sum": ortho,
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    write_result(args.out_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
