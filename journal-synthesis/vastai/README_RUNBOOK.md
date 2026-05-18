# vast.ai Runbook - blocked journal experiments (E5-E10 + full-scale E1-E3)

This package reproduces / scales every experiment that EXPERIMENTS.md
marks BLOCKED in the preparation environment (no GPU, no models, no
network). Nothing here has been run. There are NO results in this repo
(`results/` ships empty on purpose, `.gitignore`d). Every script
computes-and-writes; none hardcodes a number. No model identifier is
stored in this package or its artifacts - you pass the checkpoint.

## 0. Honesty contract

- Scripts only emit a result after real computation; `common/schema.py`
  refuses to write a result JSON that lacks the `computed_on` stamp the
  script sets ONLY after finishing real work.
- The prior CPU latency figure (4.07 us, isolated DP, CORD excluded) is
  carried forward verbatim by E8 as a contrast baseline and never
  overwritten.
- If a step has no GPU it raises and aborts (fail-fast); it never
  silently produces CPU output dressed up as the deployed number.

## 1. Recommended instance

- Single modern GPU (24 GB VRAM is comfortable; 16 GB works at
  `--batch 2`). Donut-base inference is the only heavy load; no
  training.
- ~50 GB disk (model checkpoints + CORD/SROIE/WildReceipt + decoded
  records).
- A vast.ai **PyTorch** template image (ships a matched torch +
  torchvision; do NOT pip-install torch - see requirements.txt).

## 2. One-time setup on the box

```bash
git clone <this-repo> && cd <repo>/journal-synthesis/vastai
pip install -r requirements.txt
```

That `pip install` line is exactly:

```
pip install -r requirements.txt
```

(It deliberately does NOT install torch/torchvision; use the image's.)

## 3. Fetch models + data using the PRIOR repos' existing scripts

Clone the two prior repos next to this one (read-only reference; do not
modify them) and use their existing fetchers - this package does not
re-implement data fetching:

```bash
# CORD-v2 (train+test+dev), canonical images/*.png + annotations/*.json
python /path/to/arith-gating/scripts/fetch_data.py --dataset cord
python /path/to/arith-gating/scripts/fetch_cord_dev.py        # dev split
# WildReceipt (for the natural-shift pair / breadth):
python /path/to/arith-gating/scripts/fetch_wildreceipt.py
# (SROIE Task-3, if used as the canonical shift partner, via the same
#  arith-gating fetch path / triology sroie_canonical helper.)
```

KIE checkpoint: the prior repos use a Donut CORD-v2 checkpoint from
HuggingFace; download it with `huggingface-cli download <ckpt>` and pass
its local path as `--checkpoint`. The checkpoint id is NOT stored in
this package - you supply it at run time.

Result of fetching: directories laid out as
`<root>/images/<id>.png` + `<root>/annotations/<id>.json`. Pass each as
`label=path`, e.g. `cord=/data/cord/test`.

## 4. Run

Either drive everything:

```bash
export CHECKPOINT=/data/ckpts/donut-cord
export CORD=cord=/data/cord/test
export SROIE=sroie=/data/sroie/test           # optional
export WILDRECEIPT=wildreceipt=/data/wildreceipt/test  # optional
bash run_all.sh
```

…or run one experiment at a time (each is self-contained, argparse,
deterministic seed 12345):

```bash
python3 e1e3_fullscale.py        --checkpoint $CKPT --corpus $CORD
python3 e5_integrated_benchmark.py --checkpoint $CKPT --corpora $CORD $SROIE
python3 e6_multi_shift_pairs.py  --checkpoint $CKPT --corpora $CORD $SROIE \
                                 --pairs cord:sroie
python3 e7_mechanism_synthetic_shift.py --checkpoint $CKPT --base $CORD
python3 e8_end_to_end_latency.py --checkpoint $CKPT --corpus $CORD
python3 e9_alt_verifier_bakeoff.py --checkpoint $CKPT --corpus $CORD
python3 e10_power_and_breadth.py --checkpoints $CKPT --corpora $CORD
```

## 5. Where outputs land

`vastai/results/`:

- `<EXP>.json`            - the computed result for that experiment
- `<EXP>_records.jsonl`   - the unified per-receipt records (shared
  schema: receipt_id, corpus, backbone, gold_total, pred_total,
  softmax_confidence, c_seq, arith_pass, subset_sum_verdict,
  beam_margin) so receipt_ids align by construction across experiments
- `MANIFEST.txt`          - what `run_all.sh` completed

Do NOT commit `results/` back from a run unless you have audited it; the
paper's `\pending` tags should only be replaced after a human checks
these against PREREGISTRATION.md (the excluded internal-QA step).

## 6. Per-experiment summary

| Exp | Computes | Needs |
|-----|----------|-------|
| E1E3 full-scale | E1/E2/E3 (same math as run_analysis.py) on full corpus, removing the n=100 proxy caveat | GPU, ckpt, full CORD |
| E5 | unified pipeline, four-way head-to-head at matched cost (H1) + error-decorrelation (H2) at scale | GPU, ckpt, >=1 corpus |
| E6 | Axis-B log2 variance-ratio + KS + permutation p over multiple natural shift pairs | GPU, ckpt, >=2 corpora |
| E7 | margin-variance vs independently dialed difficulty/shift (H3) + ablation hooks | GPU, ckpt, 1 base corpus |
| E8 | real end-to-end GPU latency, gate inline, vs prior CPU 4.07 us | GPU, ckpt, 1 corpus |
| E9 | line-item / subtotal+tax / rounding verifiers vs subset-sum (H4) | GPU, ckpt, corpus w/ line items |
| E10 | per-corpus/backbone precision + Wilson half-width vs target + n-needed | GPU, >=1 ckpt, >=1 corpus |

## 7. Wall-clock + cost (ESTIMATE - NOT measured)

The numbers below are an **ESTIMATE** only. They assume one modern GPU
(~USD 0.40-0.80 /hr typical vast.ai single-GPU rate), Donut-base
inference at batch 4, CORD test ~1000 receipts, ~0.3-0.6 s/receipt with
the beam-2 margin pass. They have NOT been measured (no GPU here) and
will vary with GPU, batch size, and corpus size.

| Exp | ESTIMATE wall-clock | ESTIMATE cost @ USD 0.60/hr |
|-----|---------------------|------------------------------|
| E1E3 full-scale | ~0.5-1.5 h | ~USD 0.3-0.9 |
| E5 | ~1-3 h (multi-corpus, two passes) | ~USD 0.6-1.8 |
| E6 | ~1-2 h per pair | ~USD 0.6-1.2 /pair |
| E7 | ~2-5 h (difficulty x shift grid) | ~USD 1.2-3.0 |
| E8 | ~0.3-0.7 h (timing workload) | ~USD 0.2-0.5 |
| E9 | ~0.5-1.0 h | ~USD 0.3-0.6 |
| E10 | ~0.5-1.0 h per (corpus,backbone) | ~USD 0.3-0.6 /cell |
| **Full `run_all.sh`** | **ESTIMATE ~6-14 h** | **ESTIMATE ~USD 4-9** |

Treat the table as planning guidance, not a measurement.
