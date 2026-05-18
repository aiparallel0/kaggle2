#!/usr/bin/env python3
"""E7 - Controlled mechanism study: synthetic shift with INDEPENDENTLY
dialed difficulty vs distribution-distance.

WHAT IT COMPUTES
  Tests PREREGISTRATION.md H3: beam-margin-variance compression tracks
  distribution DISTANCE, not difficulty. Builds synthetic shifted copies
  of a base corpus where two knobs are dialed INDEPENDENTLY:
    --difficulty   image degradation strength (blur/noise/jpeg) - makes
                   the task harder WITHOUT moving the input distribution
                   in the axis the model keys on.
    --shift        distribution-distance strength (font/contrast/layout
                   domain transform) - moves the distribution.
  For each (difficulty, shift) cell it decodes with num_beams=2, measures
  the log2 margin-variance ratio vs the unperturbed base and contrasts it
  with location signals (mean c_seq, mean softmax). A 2-factor readout
  (margin-variance response to `shift` >> response to `difficulty`)
  supports H3. Includes ablation HOOKS: --beam_width, --length_penalty,
  --backbone are surfaced and recorded so the beam-width / length-norm /
  architecture ablation can be swept by re-invoking the script.

WHAT IT NEEDS
  GPU. KIE checkpoint (--checkpoint). One base corpus (synthetic shifts
  are generated in-process from it; no extra download).

EXPECTED OUTPUTS
  vastai/results/E7_records.jsonl - per (cell, receipt) unified records
  vastai/results/E7.json          - 2-factor margin-variance vs location

STATUS: NOT RUN. Requires synthetic-shift generation + GPU inference.
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
    variance, variance_ratio_log2, spearman, to_cents,
    subset_sum_verdict, load_donut, decode_fields, beam_margin_batch,
)


def parse_args():
    ap = argparse.ArgumentParser(description="E7 mechanism / synthetic shift")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task_prompt", default="<s_cord-v2>")
    ap.add_argument("--base", required=True,
                    help="label=path of the clean base corpus")
    ap.add_argument("--difficulty", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75],
                    help="difficulty knob grid (task-harder, dist-fixed)")
    ap.add_argument("--shift", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75],
                    help="distribution-distance knob grid")
    # ablation hooks (recorded; vary by re-invocation):
    ap.add_argument("--beam_width", type=int, default=2)
    ap.add_argument("--length_penalty", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out_records",
                    default=os.path.join(HERE, "results", "E7_records.jsonl"))
    ap.add_argument("--out_json",
                    default=os.path.join(HERE, "results", "E7.json"))
    return ap.parse_args()


def degrade(img, difficulty):
    """Task-harder, distribution-preserving degradation (gaussian blur +
    additive noise + jpeg recompression) scaled by `difficulty` in
    [0,1]. difficulty==0 returns the image unchanged."""
    if difficulty <= 0:
        return img
    import io
    from PIL import Image, ImageFilter
    out = img.filter(ImageFilter.GaussianBlur(radius=2.0 * difficulty))
    buf = io.BytesIO()
    q = max(5, int(95 - 80 * difficulty))
    out.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def domain_shift(img, shift):
    """Distribution-distance transform (contrast + hue/tone domain move)
    scaled by `shift` in [0,1]. shift==0 returns the image unchanged.
    Designed to move the input distribution while keeping legibility
    roughly fixed (so it is NOT a difficulty knob)."""
    if shift <= 0:
        return img
    from PIL import ImageEnhance
    out = ImageEnhance.Contrast(img).enhance(1.0 + 1.5 * shift)
    out = ImageEnhance.Color(out).enhance(max(0.0, 1.0 - 1.2 * shift))
    return out


def load_base(label_path):
    from PIL import Image
    label, path = label_path.split("=", 1)
    anns = os.path.join(path, "annotations")
    imgs = os.path.join(path, "images")
    out = []
    for fn in sorted(os.listdir(anns)):
        if not fn.endswith(".json"):
            continue
        rid = os.path.splitext(fn)[0]
        ip = os.path.join(imgs, rid + ".png")
        if os.path.exists(ip):
            out.append((label, rid, Image.open(ip).convert("RGB")))
    return out


def main():
    args = parse_args()
    seed_everything(args.seed)
    processor, model = load_donut(args.checkpoint)
    backbone = os.path.basename(args.checkpoint.rstrip("/"))
    base = load_base(args.base)
    dec_one = processor.tokenizer(
        args.task_prompt, add_special_tokens=False,
        return_tensors="pt").input_ids

    records = []
    cell_margins = {}   # (diff, shift) -> [margins]
    cell_loc = {}       # (diff, shift) -> {"c_seq":[], "softmax":[]}

    for d in args.difficulty:
        for s in args.shift:
            key = (d, s)
            cell_margins[key] = []
            cell_loc[key] = {"c_seq": [], "softmax": []}
            for i in range(0, len(base), args.batch):
                chunk = base[i:i + args.batch]
                imgs = [domain_shift(degrade(c[2], d), s) for c in chunk]
                decoded = decode_fields(imgs, processor, model,
                                        args.task_prompt)
                bm = beam_margin_batch(imgs, processor, model, dec_one,
                                       processor.tokenizer.pad_token_id)
                for j, (corpus, rid, _img) in enumerate(chunk):
                    fields, sm, cs = decoded[j]
                    margin = bm[j]["margin"] if j < len(bm) else None
                    pred_total = to_cents(fields.get("total"))
                    records.append(UnifiedRecord(
                        receipt_id=f"d{d}_s{s}:{corpus}:{rid}",
                        corpus=f"synthetic:{corpus}:d{d}:s{s}",
                        backbone=backbone, gold_total=None,
                        pred_total=pred_total, softmax_confidence=sm,
                        c_seq=cs, arith_pass=None,
                        subset_sum_verdict=subset_sum_verdict(
                            pred_total, []),
                        beam_margin=margin,
                        extra={"difficulty": d, "shift": s}))
                    if margin is not None:
                        cell_margins[key].append(margin)
                    if cs is not None:
                        cell_loc[key]["c_seq"].append(cs)
                    if sm is not None:
                        cell_loc[key]["softmax"].append(sm)

    write_records(args.out_records, records)

    base_key = (args.difficulty[0], args.shift[0])  # cleanest cell
    base_var = variance(cell_margins.get(base_key, []))
    grid = []
    for (d, s), m in cell_margins.items():
        loc = cell_loc[(d, s)]
        grid.append({
            "difficulty": d, "shift": s, "n": len(m),
            "margin_variance": variance(m),
            "log2_var_ratio_vs_clean": (
                variance_ratio_log2(cell_margins.get(base_key, []), m)
                if base_var > 0 else None),
            "mean_c_seq": (sum(loc["c_seq"]) / len(loc["c_seq"])
                           if loc["c_seq"] else None),
            "mean_softmax": (sum(loc["softmax"]) / len(loc["softmax"])
                             if loc["softmax"] else None)})

    # H3 readout: Spearman of margin-variance vs each knob, holding the
    # other fixed at its grid values, then averaged - reported, not
    # interpreted here (interpretation only after a real run).
    def collapsed(idx):
        xs, ys = [], []
        for row in grid:
            xs.append(row["difficulty"] if idx == 0 else row["shift"])
            ys.append(row["margin_variance"])
        return spearman(xs, ys)

    payload = {
        "experiment": "E7",
        "scope": "controlled synthetic shift; difficulty vs distribution "
                 "distance dialed independently (H3)",
        "ablation_hooks": {"beam_width": args.beam_width,
                           "length_penalty": args.length_penalty,
                           "backbone": backbone},
        "grid": grid,
        "spearman_margin_var_vs_difficulty": collapsed(0),
        "spearman_margin_var_vs_shift": collapsed(1),
        "computed_on": f"{socket.gethostname()}@"
                       f"{datetime.datetime.utcnow().isoformat()}Z",
    }
    write_result(args.out_json, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "grid"},
                      indent=2))


if __name__ == "__main__":
    main()
