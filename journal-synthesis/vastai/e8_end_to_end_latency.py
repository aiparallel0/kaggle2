#!/usr/bin/env python3
"""E8 - Real end-to-end GPU latency with the gate INLINE.

WHAT IT COMPUTES
  The prior work disclosed a CPU-only standalone subset-sum DP latency of
  4.07 us median (triology runs/time_budget_cpu.json, explicitly an
  isolated-DP CPU number, CORD excluded, NOT fabricated). E8 measures the
  TRUE deployed cost: full GPU KIE decode + Axis-A subset-sum + Axis-B
  beam-margin gate, all inline, per receipt. It reports the end-to-end
  wall-clock distribution AND the marginal added cost of the gate
  (decode-only vs decode+gate), using the SAME timing methodology as
  triology (perf_counter around the exact code path, warmup discarded,
  per-receipt percentiles, CUDA synchronize before each stop). The 4.07us
  CPU figure is carried forward verbatim as the contrast baseline; the
  script never overwrites it and never invents a GPU number - it measures.

WHAT IT NEEDS
  GPU. KIE checkpoint (--checkpoint). One corpus for the timing workload.

EXPECTED OUTPUTS
  vastai/results/E8.json - end-to-end + decode-only + gate-marginal
                           latency percentiles (us/ms), warmup discarded

STATUS: NOT RUN. No GPU in the prep environment; measuring here would be
fabrication. Computes-and-writes a measured distribution only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from common import (  # noqa: E402
    write_result, seed_everything, median, to_cents,
    subset_sum_verdict, load_donut, decode_fields, beam_margin_batch,
)

# Carried-forward prior-work CPU baseline (verbatim; NOT recomputed here).
PRIOR_CPU_DP_MEDIAN_US = 4.07
PRIOR_CPU_NOTE = ("triology runs/time_budget_cpu.json: standalone CPU "
                  "subset-sum DP, isolated, CORD excluded - prior work, "
                  "carried forward for contrast only.")


def parse_args():
    ap = argparse.ArgumentParser(description="E8 end-to-end GPU latency")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--corpus", required=True, help="label=path")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--max_receipts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "E8.json"))
    return ap.parse_args()


def cuda_sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def percentiles(xs):
    s = sorted(xs)
    if not s:
        return {}

    def q(p):
        return s[min(len(s) - 1, int(p * len(s)))]

    return {"n": len(s), "mean": sum(s) / len(s), "median": median(s),
            "p50": q(0.50), "p95": q(0.95), "p99": q(0.99),
            "min": s[0], "max": s[-1]}


def main():
    args = parse_args()
    seed_everything(args.seed)
    processor, model = load_donut(args.checkpoint)
    dec_one = processor.tokenizer(
        args.task_prompt, add_special_tokens=False,
        return_tensors="pt").input_ids
    pad = processor.tokenizer.pad_token_id

    from PIL import Image
    label, path = args.corpus.split("=", 1)
    anns = os.path.join(path, "annotations")
    imgs_dir = os.path.join(path, "images")
    rids = [os.path.splitext(f)[0]
            for f in sorted(os.listdir(anns)) if f.endswith(".json")]
    rids = rids[:args.max_receipts]

    decode_us, e2e_us, gate_us = [], [], []
    for n, rid in enumerate(rids):
        ip = os.path.join(imgs_dir, rid + ".png")
        if not os.path.exists(ip):
            continue
        img = [Image.open(ip).convert("RGB")]

        cuda_sync()
        t0 = time.perf_counter()
        decoded = decode_fields(img, processor, model, args.task_prompt)
        cuda_sync()
        t1 = time.perf_counter()
        bm = beam_margin_batch(img, processor, model, dec_one, pad)
        cuda_sync()
        t2 = time.perf_counter()
        fields = decoded[0][0]
        pred_total = to_cents(fields.get("total"))
        t3 = time.perf_counter()
        _ = subset_sum_verdict(pred_total, [])
        _ = bm[0]["margin"] if bm else None
        t4 = time.perf_counter()

        if n < args.warmup:
            continue  # discard warmup iterations
        decode_us.append((t1 - t0) * 1e6)
        gate_us.append(((t2 - t1) + (t4 - t3)) * 1e6)
        e2e_us.append((t4 - t0) * 1e6)

    payload = {
        "experiment": "E8",
        "scope": "real end-to-end GPU latency, gate inline; warmup "
                 f"({args.warmup}) discarded",
        "decode_only_us": percentiles(decode_us),
        "gate_marginal_us": percentiles(gate_us),
        "end_to_end_us": percentiles(e2e_us),
        "prior_cpu_dp_median_us": PRIOR_CPU_DP_MEDIAN_US,
        "prior_cpu_note": PRIOR_CPU_NOTE,
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    write_result(args.out_json, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
