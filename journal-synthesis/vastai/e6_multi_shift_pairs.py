#!/usr/bin/env python3
"""E6 - Axis-B variance-ratio + significance battery over MULTIPLE
natural shift pairs.

WHAT IT COMPUTES
  For each (in-dist corpus, shifted corpus) pair in --pairs, decodes both
  sides with num_beams=2 and extracts the beam-margin per receipt
  (arith-gating BM_beam_margin method, reused via common.pipeline), then
  reports the log2 variance ratio  log2( Var[in] / Var[shifted] )  (the
  Paper-2 compression statistic, sign convention identical to
  arith-gating BM2_compression.py), the two-sample KS distance on the
  margin distributions, and a permutation significance test on the
  variance-ratio statistic. Pairs are PARAMETERISED (--pairs), not pooled
  away (PREREGISTRATION.md analysis plan: multiple pairs reported
  individually).

WHAT IT NEEDS
  GPU. KIE checkpoint (--checkpoint). >=2 corpora forming the natural
  shift pairs, fetched via the prior repos' fetch scripts.

EXPECTED OUTPUTS
  vastai/results/E6_records.jsonl  - unified per-receipt records (every
                                     decoded receipt, both sides)
  vastai/results/E6.json           - per-pair variance ratio / KS / p

STATUS: NOT RUN. Needs new corpora + model inference (no network/GPU in
the prep environment). Computes-and-writes; never hardcodes.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import random
import socket
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (  # noqa: E402
    UnifiedRecord, write_records, write_result, seed_everything,
    variance, variance_ratio_log2, to_cents, subset_sum_verdict,
    load_donut, decode_fields, beam_margin_batch,
)


def parse_args():
    ap = argparse.ArgumentParser(description="E6 multi natural-shift pairs")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="label=path entries for every corpus referenced "
                         "by --pairs")
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="in_label:shift_label entries, e.g. "
                         "cord:sroie cord:wildreceipt")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--perm_iters", type=int, default=20000)
    ap.add_argument("--out_records",
                    default=os.path.join(HERE, "results", "E6_records.jsonl"))
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "E6.json"))
    return ap.parse_args()


def ks_distance(a, b):
    """Two-sample KS statistic (scipy if present, else exact stdlib)."""
    try:
        from scipy.stats import ks_2samp
        return float(ks_2samp(a, b).statistic)
    except Exception:
        xs = sorted(set(a) | set(b))
        na, nb = len(a), len(b)
        if na == 0 or nb == 0:
            return float("nan")
        d = 0.0
        sa = sorted(a)
        sb = sorted(b)
        for x in xs:
            fa = sum(1 for v in sa if v <= x) / na
            fb = sum(1 for v in sb if v <= x) / nb
            d = max(d, abs(fa - fb))
        return d


def perm_p_varratio(in_vals, shift_vals, iters, seed):
    """Two-sided permutation p on |log2 variance ratio| by relabelling."""
    obs = abs(variance_ratio_log2(in_vals, shift_vals))
    if obs != obs:  # nan
        return float("nan")
    pool = list(in_vals) + list(shift_vals)
    n_in = len(in_vals)
    rng = random.Random(seed)
    ge = 0
    for _ in range(iters):
        rng.shuffle(pool)
        s = abs(variance_ratio_log2(pool[:n_in], pool[n_in:]))
        if s == s and s >= obs - 1e-12:
            ge += 1
    return (ge + 1) / (iters + 1)


def decode_corpus(label, path, processor, model, args, backbone):
    from PIL import Image
    anns = os.path.join(path, "annotations")
    imgs_dir = os.path.join(path, "images")
    items = sorted(f for f in os.listdir(anns) if f.endswith(".json"))
    out_records, margins_by_id = [], {}
    dec_one = processor.tokenizer(
        args.task_prompt, add_special_tokens=False,
        return_tensors="pt").input_ids
    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        imgs, rids = [], []
        for fn in chunk:
            rid = os.path.splitext(fn)[0]
            ip = os.path.join(imgs_dir, rid + ".png")
            if not os.path.exists(ip):
                continue
            imgs.append(Image.open(ip).convert("RGB"))
            rids.append(rid)
        if not imgs:
            continue
        decoded = decode_fields(imgs, processor, model, args.task_prompt)
        bm = beam_margin_batch(imgs, processor, model, dec_one,
                               processor.tokenizer.pad_token_id)
        for j, rid in enumerate(rids):
            fields, sm, cs = decoded[j]
            margin = bm[j]["margin"] if j < len(bm) else None
            pred_total = to_cents(fields.get("total"))
            rec = UnifiedRecord(
                receipt_id=f"{label}:{rid}", corpus=label,
                backbone=backbone, gold_total=None,
                pred_total=pred_total, softmax_confidence=sm, c_seq=cs,
                arith_pass=None,
                subset_sum_verdict=subset_sum_verdict(pred_total, []),
                beam_margin=margin)
            out_records.append(rec)
            if margin is not None:
                margins_by_id[rec.receipt_id] = margin
    return out_records, [margins_by_id[k] for k in sorted(margins_by_id)]


def main():
    args = parse_args()
    seed_everything(args.seed)
    processor, model = load_donut(args.checkpoint)
    backbone = os.path.basename(args.checkpoint.rstrip("/"))
    paths = dict(lp.split("=", 1) for lp in args.corpora)

    all_records, margins = [], {}
    for label, path in paths.items():
        recs, m = decode_corpus(label, path, processor, model, args, backbone)
        all_records.extend(recs)
        margins[label] = m
    write_records(args.out_records, all_records)

    pair_results = {}
    for spec in args.pairs:
        a, b = spec.split(":", 1)
        in_m, sh_m = margins.get(a, []), margins.get(b, [])
        pair_results[spec] = {
            "n_in": len(in_m), "n_shift": len(sh_m),
            "var_in": variance(in_m), "var_shift": variance(sh_m),
            "log2_variance_ratio": variance_ratio_log2(in_m, sh_m),
            "ks_distance_margin": ks_distance(in_m, sh_m),
            "permutation_p_two_sided": perm_p_varratio(
                in_m, sh_m, args.perm_iters, args.seed),
        }

    payload = {
        "experiment": "E6",
        "scope": "Axis-B variance-ratio + significance over multiple "
                 "natural shift pairs; pairs reported individually",
        "pairs": pair_results,
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    write_result(args.out_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
