#!/usr/bin/env python3
"""E10 - Power-resolving per-corpus replication + broadened breadth hooks.

WHAT IT COMPUTES
  Per-corpus (and per-backbone) replication of the composed-policy
  precision, decoding each corpus through the unified pipeline and
  reporting precision with a Wilson 95% CI, then checking whether the CI
  HALF-WIDTH meets a pre-registered target (--ci_halfwidth_target). For
  corpora that do not yet meet the target it computes the additional n
  required (normal-approx sample-size solve) so the user knows how much
  more data to fetch - it does NOT pad with synthetic receipts. Results
  are reported INDIVIDUALLY per corpus/backbone (PREREGISTRATION.md: not
  pooled away). --corpora and --checkpoints are lists, so the broadened-
  corpora / broadened-backbone sweep is just a longer invocation (the
  breadth "hooks").

WHAT IT NEEDS
  GPU. One or more KIE checkpoints (--checkpoints). One or more corpora
  fetched via the prior repos' fetch scripts.

EXPECTED OUTPUTS
  vastai/results/E10_records.jsonl - unified per-receipt records
  vastai/results/E10.json          - per (corpus,backbone) precision,
                                     Wilson CI, half-width vs target,
                                     additional-n-needed

STATUS: NOT RUN. Needs new data + multi-backbone GPU inference.
Computes-and-writes; never hardcodes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (  # noqa: E402
    UnifiedRecord, write_records, write_result, seed_everything,
    wilson, median, to_cents, subset_sum_verdict,
    decode_or_load,
)


def parse_args():
    ap = argparse.ArgumentParser(description="E10 power + breadth")
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="one or more KIE checkpoints (broadened-backbone "
                         "hook); ids not stored in artifacts")
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="label=path entries (broadened-corpora hook)")
    ap.add_argument("--ci_halfwidth_target", type=float, default=0.05,
                    help="pre-registered Wilson CI half-width target")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_records",
                    default=os.path.join(HERE, "results", "E10_records.jsonl"))
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "E10.json"))
    return ap.parse_args()


def required_n(p_hat, target_hw, z=1.96):
    """Normal-approx n for a target CI half-width at observed p_hat."""
    p = min(max(p_hat, 1e-6), 1 - 1e-6)
    return int(math.ceil((z * z * p * (1 - p)) / (target_hw * target_hw)))


def decode_one_corpus(label_path, ckpt, args, backbone):
    """Decode-once shared cache (identical decode_fields +
    beam_margin_batch; loaded if already produced, NO model/GPU on a
    cache hit). Per-receipt record math below is byte-for-byte the
    original, only sourced from the shared cache. The cache key includes
    the checkpoint hash so each backbone gets its own records file."""
    label, _path = label_path.split("=", 1)
    primitives = decode_or_load(
        label_path, ckpt, args.task_prompt, args.batch)
    recs = []
    for p in primitives:
        rid = p["receipt_id"]
        fields = p["fields"] if isinstance(p["fields"], dict) else {}
        sm, cs = p["softmax_confidence"], p["c_seq"]
        gt = p["gold"]
        pred_total = to_cents(fields.get("total"))
        items = []
        menu = fields.get("menu")
        if isinstance(menu, list):
            for it in menu:
                if isinstance(it, dict):
                    c = to_cents(it.get("price"))
                    if c is not None:
                        items.append(c)
        tau = to_cents(fields.get("tax")) or 0
        ss = subset_sum_verdict(pred_total, items, tau)
        gp = (gt.get("gt_parse", gt) if isinstance(gt, dict) else {})
        recs.append(UnifiedRecord(
            receipt_id=f"{backbone}:{label}:{rid}", corpus=label,
            backbone=backbone, gold_total=to_cents(gp.get("total")),
            pred_total=pred_total, softmax_confidence=sm, c_seq=cs,
            arith_pass=(ss == "pass"), subset_sum_verdict=ss,
            beam_margin=p["beam_margin"],
            extra={"correct": fields == gp}))
    return recs


def main():
    args = parse_args()
    seed_everything(args.seed)
    all_records = []
    cells = {}
    for ckpt in args.checkpoints:
        backbone = os.path.basename(ckpt.rstrip("/"))
        for lp in args.corpora:
            label, _path = lp.split("=", 1)
            recs = decode_one_corpus(lp, ckpt, args, backbone)
            all_records.extend(recs)
            cells[(backbone, label)] = recs

    write_records(args.out_records, all_records)

    per_cell = {}
    for (backbone, label), recs in cells.items():
        c_seqs = [r.c_seq for r in recs if r.c_seq is not None]
        margins = [r.beam_margin for r in recs if r.beam_margin is not None]
        c_thr = median(c_seqs)
        m_thr = median(margins) if margins else None
        composed = [r for r in recs
                    if r.arith_pass is True and r.beam_margin is not None
                    and m_thr is not None and r.beam_margin >= m_thr]
        n = len(composed)
        corr = sum(1 for r in composed if r.extra.get("correct"))
        lo, hi = wilson(corr, n) if n else (None, None)
        p_hat = (corr / n) if n else None
        hw = ((hi - lo) / 2.0) if (lo is not None and hi is not None) \
            else None
        per_cell[f"{backbone}|{label}"] = {
            "n_total": len(recs), "n_composed_accept": n,
            "n_correct": corr, "precision": p_hat,
            "precision_wilson95": [lo, hi],
            "ci_halfwidth": hw,
            "meets_target": (hw is not None
                             and hw <= args.ci_halfwidth_target),
            "n_needed_for_target": (
                required_n(p_hat, args.ci_halfwidth_target)
                if p_hat is not None else None),
            "c_seq_threshold_median": c_thr,
            "beam_margin_threshold_median": m_thr}

    payload = {
        "experiment": "E10",
        "scope": "power-resolving per-corpus/backbone replication; "
                 "reported individually, not pooled",
        "ci_halfwidth_target": args.ci_halfwidth_target,
        "per_corpus_backbone": per_cell,
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    write_result(args.out_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
