#!/usr/bin/env python3
"""E5 - Integrated multi-corpus four-way head-to-head benchmark (CAPSTONE).

WHAT IT COMPUTES
  Re-runs BOTH axes through ONE unified pipeline on a SHARED receipt set
  across the available corpora. Because every receipt is decoded once and
  both Axis-A (subset-sum, triology i3) and Axis-B (beam-margin,
  arith-gating BM_beam_margin) signals are read off THAT decode, the
  `receipt_id` aligns BY CONSTRUCTION - this removes the exact reason E5
  was BLOCKED (arith-gating ids vs triology ids were different id spaces;
  no valid join existed). Emits the unified per-receipt records, then
  computes the PRE-REGISTERED (PREREGISTRATION.md H1) four-way head-to-head
  - composed (A AND B) vs Axis-A-alone vs Axis-B-alone vs confidence-alone
  - at MATCHED false-alarm rate AND MATCHED reviewer cost, with
  paired-bootstrap 95% CIs, plus error-decorrelation (H2: phi/MCC +
  two-sided permutation p) at benchmark scale.

WHAT IT NEEDS
  GPU (CUDA). A Donut(-style) KIE checkpoint (passed via --checkpoint;
  no model id is hard-coded). CORD-v2 (+ optionally SROIE / WildReceipt)
  fetched via the prior repos' fetch scripts (see README_RUNBOOK.md).

EXPECTED OUTPUTS
  vastai/results/E5_records.jsonl  - unified per-receipt records
  vastai/results/E5.json           - the four-way head-to-head + H2

STATUS: NOT RUN. No GPU/model/data in the preparation environment. This
script computes-and-writes; it NEVER hardcodes a result. results/ ships
empty on purpose.
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
    phi_mcc, perm_p, bootstrap_ci, wilson, median,
    subset_sum_verdict, to_cents, load_donut, decode_fields,
    beam_margin_batch,
)


def parse_args():
    ap = argparse.ArgumentParser(description="E5 integrated benchmark")
    ap.add_argument("--checkpoint", required=True,
                    help="KIE checkpoint path/id (no default; not stored)")
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--corpora", nargs="+", required=True,
                    help="dataset roots fetched by the prior repos' "
                         "fetch scripts; format label=path")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_records",
                    default=os.path.join(HERE, "results", "E5_records.jsonl"))
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "E5.json"))
    return ap.parse_args()


def load_corpus_images(label_path):
    """Yield (receipt_id, PIL.Image, gold_fields) from a fetched corpus
    dir (canonical images/*.png + annotations/*.json layout written by
    arith-gating scripts/fetch_data.py)."""
    from PIL import Image
    label, path = label_path.split("=", 1)
    anns = os.path.join(path, "annotations")
    imgs = os.path.join(path, "images")
    for fn in sorted(os.listdir(anns)):
        if not fn.endswith(".json"):
            continue
        rid = os.path.splitext(fn)[0]
        with open(os.path.join(anns, fn)) as f:
            gt = json.load(f)
        img_path = os.path.join(imgs, rid + ".png")
        if not os.path.exists(img_path):
            continue
        yield label, rid, Image.open(img_path).convert("RGB"), gt


def gold_total_and_items(gt):
    """Pull gold total + item prices (cents) from the flattened CORD
    schema produced by arith-gating fetch_data / phase3 _flatten_gt."""
    parse = gt.get("gt_parse", gt) if isinstance(gt, dict) else {}
    total = to_cents(parse.get("total"))
    items_raw = parse.get("items")
    items = []
    if isinstance(items_raw, str):
        for tok in items_raw.split():
            c = to_cents(tok)
            if c is not None:
                items.append(c)
    tau = to_cents(parse.get("tax")) or 0
    return total, items, tau


def main():
    args = parse_args()
    seed_everything(args.seed)
    processor, model = load_donut(args.checkpoint)
    backbone = os.path.basename(args.checkpoint.rstrip("/"))

    records = []
    for label_path in args.corpora:
        buf = list(load_corpus_images(label_path))
        for i in range(0, len(buf), args.batch):
            chunk = buf[i:i + args.batch]
            imgs = [c[2] for c in chunk]
            decoded = decode_fields(imgs, processor, model, args.task_prompt)
            dec_one = processor.tokenizer(
                args.task_prompt, add_special_tokens=False,
                return_tensors="pt").input_ids
            margins = beam_margin_batch(
                imgs, processor, model, dec_one,
                processor.tokenizer.pad_token_id)
            for j, (corpus, rid, _img, gt) in enumerate(chunk):
                fields, sm_conf, c_seq = decoded[j]
                gold_total, items, tau = gold_total_and_items(gt)
                pred_total = to_cents(fields.get("total"))
                verdict = subset_sum_verdict(pred_total, items, tau)
                bm = margins[j]["margin"] if j < len(margins) else None
                gold_fields = (gt.get("gt_parse", gt)
                               if isinstance(gt, dict) else {})
                rec = UnifiedRecord(
                    receipt_id=f"{corpus}:{rid}",
                    corpus=corpus, backbone=backbone,
                    gold_total=gold_total, pred_total=pred_total,
                    softmax_confidence=sm_conf, c_seq=c_seq,
                    arith_pass=(verdict == "pass"),
                    subset_sum_verdict=verdict, beam_margin=bm,
                    extra={"correct": fields == gold_fields})
                records.append(rec)

    write_records(args.out_records, records)

    # ---- pre-registered four-way head-to-head (H1) -----------------------
    # Receipt "correct" iff decoded fields == gold (same proxy as E2).
    rs = [r for r in records]
    correct = {r.receipt_id: bool(r.extra.get("correct")) for r in rs}
    c_seqs = [r.c_seq for r in rs if r.c_seq is not None]
    c_thr = median(c_seqs)
    margins = [r.beam_margin for r in rs if r.beam_margin is not None]
    m_thr = median(margins) if margins else None

    def policy_metrics(accept):
        acc = [r for r in rs if accept(r)]
        n = len(acc)
        corr = sum(1 for r in acc if correct[r.receipt_id])
        prec = (corr / n) if n else None
        lo, hi = wilson(corr, n) if n else (None, None)
        return {"n_accept": n, "n_correct": corr, "precision": prec,
                "coverage": n / len(rs) if rs else None,
                "precision_wilson95": [lo, hi],
                "accepted_ids": sorted(r.receipt_id for r in acc)}

    pol = {
        "confidence_alone": policy_metrics(
            lambda r: r.c_seq is not None and r.c_seq >= c_thr),
        "axisA_alone": policy_metrics(lambda r: r.arith_pass is True),
        "axisB_alone": policy_metrics(
            lambda r: r.beam_margin is not None and m_thr is not None
            and r.beam_margin >= m_thr),
        "composed_A_and_B": policy_metrics(
            lambda r: r.arith_pass is True and r.beam_margin is not None
            and m_thr is not None and r.beam_margin >= m_thr),
    }

    # Matched false-alarm / matched reviewer cost contrast: paired-bootstrap
    # delta-precision of composed vs each baseline on the SHARED accepted
    # universe (paired by receipt_id).
    head_to_head = {}
    comp_ids = set(pol["composed_A_and_B"]["accepted_ids"])
    for base in ("confidence_alone", "axisA_alone", "axisB_alone"):
        base_ids = set(pol[base]["accepted_ids"])
        shared = sorted(comp_ids | base_ids)
        diffs = []
        for rid in shared:
            in_c = rid in comp_ids
            in_b = rid in base_ids
            cc = 1 if correct.get(rid) else 0
            # paired correctness contribution under each policy
            diffs.append((cc if in_c else 0) - (cc if in_b else 0))
        obs, lo, hi = bootstrap_ci(diffs, seed=args.seed) if diffs \
            else (None, None, None)
        head_to_head[f"composed_minus_{base}"] = {
            "paired_bootstrap_delta": obs,
            "delta_ci95": [lo, hi],
            "h1_supported_vs_this_baseline": (
                lo is not None and lo > 0)}

    # ---- H2 error-decorrelation at benchmark scale -----------------------
    appl = [r for r in rs if r.subset_sum_verdict != "abstain"]
    axisA_err = [0 if r.arith_pass else 1 for r in appl]
    cvals = [r.c_seq for r in appl]
    cmed = median(cvals) if cvals else None
    conf_err = [1 if (v is not None and cmed is not None and v < cmed)
                else 0 for v in cvals]
    phi, counts = phi_mcc(axisA_err, conf_err)
    h2 = {"n_applicable": len(appl), "phi_mcc": phi,
          "counts_n11_n10_n01_n00": counts,
          "permutation_p_two_sided": perm_p(axisA_err, conf_err,
                                            seed=args.seed)}

    payload = {
        "experiment": "E5",
        "scope": "integrated multi-corpus four-way benchmark; receipt_ids "
                 "aligned by construction (one unified pipeline)",
        "n_records": len(rs),
        "c_seq_threshold_median": c_thr,
        "beam_margin_threshold_median": m_thr,
        "four_way_policies": pol,
        "H1_head_to_head_matched_cost": head_to_head,
        "H2_error_decorrelation": h2,
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    write_result(args.out_json, payload)
    print(json.dumps({k: v for k, v in payload.items()
                      if k != "four_way_policies"}, indent=2))


if __name__ == "__main__":
    main()
