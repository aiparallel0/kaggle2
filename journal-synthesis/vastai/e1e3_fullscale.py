#!/usr/bin/env python3
"""E1/E2/E3 at FULL corpus scale (not the n=100 CORD proxy).

WHAT IT COMPUTES
  The exact same three analyses as experiments/run_analysis.py (E1
  error-decorrelation phi/MCC + permutation p; E2 confidence-alone vs
  Axis-A-alone vs composed precision/coverage + blind-spots; E3
  precision-coverage frontier sweep) but on the FULL fetched corpus
  decoded through the unified pipeline, removing the GLOBAL SCOPE WARNING
  (CORD-only, n=100, single pipeline) stated in EXPERIMENTS.md. Metric
  math is reused verbatim from common (which is itself a verbatim port of
  run_analysis.py), so a full-scale number is directly comparable to the
  n=100 proxy.

WHAT IT NEEDS
  GPU. KIE checkpoint (--checkpoint). The full corpus (all splits)
  fetched via the prior repos' fetch scripts.

EXPECTED OUTPUTS
  vastai/results/E1E3_records.jsonl - full-scale unified records
  vastai/results/E1E3.json          - E1/E2/E3 at full scale

STATUS: NOT RUN. Needs full corpus + GPU. Computes-and-writes; never
hardcodes. (Schema mirrors experiments/results_E1_E4.json E1/E2/E3.)
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
    phi_mcc, perm_p, median, to_cents, subset_sum_verdict,
    decode_or_load,
)


def parse_args():
    ap = argparse.ArgumentParser(description="E1-E3 full scale")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--corpus", required=True,
                    help="label=path of the FULL corpus (all splits)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_records",
                    default=os.path.join(HERE, "results",
                                         "E1E3_records.jsonl"))
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "E1E3.json"))
    return ap.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    # Decode-once shared cache: identical decode (decode_fields greedy +
    # beam_margin) but loaded from results/<label>__<ckpthash>.records.jsonl
    # if already produced (NO model / NO GPU on a cache hit). The metric
    # math below is byte-for-byte the original; only the per-receipt
    # primitives (fields, c_seq, softmax, gold) are now sourced from the
    # shared cache instead of an inline re-decode.
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
        records.append(UnifiedRecord(
            receipt_id=f"{label}:{rid}", corpus=label,
            backbone=backbone, gold_total=to_cents(gp.get("total")),
            pred_total=pred_total, softmax_confidence=sm, c_seq=cs,
            arith_pass=(ss == "pass"), subset_sum_verdict=ss,
            beam_margin=None,
            extra={"correct": fields == gp,
                   "n_applicable": 0 if ss == "abstain" else 1}))

    write_records(args.out_records, records)
    rs = records
    n = len(rs)

    # ---- E1 (same definitions as run_analysis.py) -----------------------
    appl = [r for r in rs if r.subset_sum_verdict != "abstain"]
    axisA_err = [0 if r.arith_pass else 1 for r in appl]

    def conf_block(getter):
        vals = [getter(r) for r in appl]
        med = median(vals) if vals else None
        ev = [1 if (v is not None and med is not None and v < med) else 0
              for v in vals]
        phi, counts = phi_mcc(axisA_err, ev)
        return {"median_split": med, "phi_mcc": phi,
                "counts_n11_n10_n01_n00": counts,
                "permutation_p_two_sided": perm_p(axisA_err, ev,
                                                  seed=args.seed)}

    e1 = {
        "scope": "FULL-SCALE (not the n=100 CORD proxy)",
        "n_records": n, "n_applicable_gt0": len(appl),
        "n_applicable_eq0_excluded_from_axisA": n - len(appl),
        "axisA_error_rate_on_applicable": (
            sum(axisA_err) / len(appl) if appl else None),
        "cseq_proxy": conf_block(lambda r: r.c_seq),
        "softmax_proxy": conf_block(lambda r: r.softmax_confidence),
    }

    # ---- E2 -------------------------------------------------------------
    def correct(r):
        return bool(r.extra.get("correct"))

    c_all = [r.c_seq for r in rs if r.c_seq is not None]
    thr = median(c_all) if c_all else None

    def policy(accept):
        acc = [r for r in rs if accept(r)]
        k = len(acc)
        corr = sum(1 for r in acc if correct(r))
        return {"n_accept": k, "n_correct": corr,
                "precision": (corr / k) if k else None,
                "coverage": k / n if n else None,
                "ids": sorted(r.receipt_id for r in acc)}

    co = policy(lambda r: r.c_seq is not None and thr is not None
                and r.c_seq >= thr)
    aa = policy(lambda r: r.arith_pass is True)
    cp = policy(lambda r: r.arith_pass is True and r.c_seq is not None
                and thr is not None and r.c_seq >= thr)
    corr_ids = {r.receipt_id for r in rs if correct(r)}
    co_ids, aa_ids, cp_ids = (set(co.pop("ids")), set(aa.pop("ids")),
                              set(cp.pop("ids")))
    e2 = {
        "scope": "FULL-SCALE",
        "c_seq_threshold_median": thr,
        "confidence_alone": co, "axisA_alone": aa,
        "composed_axisA_AND_cseq": cp,
        "blindspot_conf_wrong_accepts_caught_by_composition":
            len((co_ids - corr_ids) - cp_ids),
        "blindspot_axisA_wrong_accepts_caught_by_composition":
            len((aa_ids - corr_ids) - cp_ids),
    }

    # ---- E3 frontier ----------------------------------------------------
    frontier = []
    for t in [i / 100.0 for i in range(0, 101, 5)]:
        def pc(sel):
            k = len(sel)
            c = sum(1 for r in sel if correct(r))
            return {"coverage": k / n if n else None,
                    "precision": (c / k) if k else None, "n_accept": k}
        cf = [r for r in rs if r.c_seq is not None and r.c_seq >= t]
        cm = [r for r in rs if r.arith_pass is True
              and r.c_seq is not None and r.c_seq >= t]
        frontier.append({"c_seq_thr": t, "confidence_alone": pc(cf),
                          "composed": pc(cm)})

    payload = {
        "experiment": "E1E3_fullscale",
        "E1": e1, "E2": e2,
        "E3": {"scope": "FULL-SCALE", "frontier": frontier},
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    write_result(args.out_json, payload)
    print(json.dumps({"E1": e1, "E2": e2}, indent=2))


if __name__ == "__main__":
    main()
