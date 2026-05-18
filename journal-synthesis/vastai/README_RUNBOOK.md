# vast.ai Runbook - blocked journal experiments (E5-E10 + E1-E3)

> SCOPE HONESTY: "full-scale" is NOT automatic. Each experiment's
> emitted `scope` string now states the TRUE corpus label, the decode
> path, and the actual record count `n`. If you point `--corpus` at the
> n=100 OCR-derived CORD *validation* split (`fetch_cord_dev.py`'s
> `dev`), the scope string will say exactly that -- it will NOT claim
> full-scale. To actually run at scale, pass the full `test` (or
> train+test) split path. WildReceipt images are `.jpeg` (not `.png`);
> the loader resolves images via each annotation's own
> `image_filename`, so the corpus no longer silently collapses to
> 0 records.

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
| E1E3 | E1/E2/E3 (same math as run_analysis.py) on whatever `--corpus` points at; the JSON `scope` states the true corpus/path/n (NOT auto "full-scale") | GPU, ckpt, CORD |
| E5 | unified pipeline, four-way head-to-head at matched cost (H1) + error-decorrelation (H2); JSON `scope` lists the ACTUAL corpora + per-corpus n and flags SINGLE vs MULTI-CORPUS | GPU, ckpt, >=1 corpus |
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

## 6. Decode-once shared cache + fresh-instance runner (added)

`bootstrap.sh` is idempotent fresh-instance setup; `run_parallel.sh` is a
resumable, **decode-once** scheduler.

THE COST FIX (honest scope). Previously every experiment independently
loaded the KIE model and RE-DECODED the same corpora, so a single run
paid for the SAME Donut inference about **7x** (the cache-consuming
experiments alone: E1E3, E5, E6, E9, E10 each re-decoded the corpus, plus
the unbatched default). On a paid GPU that is ~90% wasted spend.

Now `common/records.py::decode_or_load` decodes a corpus EXACTLY as the
scripts always did (same `decode_fields` greedy pass and same
`beam_margin_batch` num_beams=2 pass, same fp16, same task prompt, same
money/cents parsing downstream) but writes the per-receipt primitives to

```
results/<corpuslabel>__<sha1(checkpoint)[:12]>.records.jsonl
```

The cache key is the **corpus label + path AND a checkpoint hash**, so a
different checkpoint can never silently reuse another's decodes. The file
carries a header with `computed_on`, `n_records`, `checkpoint_sha`, the
corpus label/path, the task prompt and a schema version; it is reused
only if the header matches AND the body line count equals `n_records`
(a truncated / half-written cache is detected and rebuilt, never
half-used). On a cache hit NO model is loaded and the GPU is never
touched.

`run_parallel.sh` therefore runs in stages:

- **Stage A (GPU, ONE pass):** the shared decode runs ONCE per distinct
  corpus, producing the `.records.jsonl` cache. This is the only GPU
  work for the cache-consuming experiments.
- **Stage B-cpu (CPU, parallel up to nproc):** E1E3 / E5 / E6 / E9 / E10
  read the cache (no model, no GPU) and only run their analysis math, so
  they are CPU-bound and run concurrently.
- **Stage B-gpu (GPU, sequential):** E7 (synthetic per-cell perturbed
  decodes) and E8 (per-receipt latency MEASUREMENT) genuinely need the
  GPU and are NOT cache consumers - their decode is real, unique work,
  not the redundant re-decode the cache removes - so they run on the
  single GPU after Stage A. Their metric math is unchanged.

This removes the ~7x redundant decode. It does NOT claim multi-GPU
scaling: there is one GPU; the win is decode-once + CPU-parallel
analyses + resumability. Every experiment's metric math, hypotheses,
seeds and outputs are byte-for-byte unchanged - the same numbers, just
sourced from the shared cache instead of an inline re-decode. All
scripts accept `--batch` (default 16, was an unbatched/`4` default);
`run_parallel.sh` passes `--batch "${BATCH:-16}"` to every experiment.

Resumable: re-running skips any experiment whose `results/<EXP>.json`
already has a real `computed_on`, and a complete Stage-A cache is reused
(a truncated one is rebuilt).

Copy-paste on a fresh vast.ai PyTorch instance. IMPORTANT: replace the
two placeholders with REAL values (no angle brackets), keep each
`export` on its own line, and set a GitHub token because arith-gating /
triology are private:

```bash
export GITHUB_TOKEN=ghp_replace_with_a_real_PAT
export CKPT_ID=naver-clova-ix/donut-base-finetuned-cord-v2
export REPO_URL=https://github.com/aiparallel0/kaggle2.git
export ARITH_URL=https://github.com/aiparallel0/arith-gating.git
export TRIOLOGY_URL=https://github.com/aiparallel0/triology.git
cd /workspace 2>/dev/null || cd ~
git clone --branch claude/prepare-papers-repos-4LUdJ --single-branch \
  "https://x-access-token:$GITHUB_TOKEN@github.com/aiparallel0/kaggle2.git" kaggle2 \
  || git -C kaggle2 pull --rebase
cd kaggle2/journal-synthesis/vastai
bash bootstrap.sh
source .env.sh
[ -n "$CHECKPOINT" ] || export CHECKPOINT=/path/to/ckpt
BATCH=16 bash run_parallel.sh
```

On the FIRST run this does ONE GPU decode pass per corpus (Stage A),
then the cache-consuming analyses (E1E3/E5/E6/E9/E10) run CPU-only and
in parallel off that cache; E7/E8 then run the genuine remaining GPU
work. A second run reuses the cache (no GPU) and skips any experiment
that already produced a real result. Tune throughput with `BATCH`
(e.g. `BATCH=8` for a 16 GB GPU, `BATCH=32` for 24 GB+).

The `CKPT_ID` above is an EXAMPLE public Donut CORD-v2 id; substitute the
checkpoint you intend to evaluate (it is never stored in the repo). The
prior fetch scripts take NO `--out`; bootstrap calls them with their
real signatures and they write into `arith-gating/data/{cord,wild}` -
`.env.sh` is filled from the directories that actually appear. SROIE has
no fetcher here; add `export SROIE=sroie=/path` yourself if you use it.

Resumable: re-running `run_parallel.sh` skips any experiment whose
`results/<EXP>.json` already has a real `computed_on` stamp; failed jobs
are logged to `results/<EXP>.log` and timed in
`results/PARALLEL_TIMING.tsv`. Nothing is faked: a job that cannot reach
a GPU aborts and is recorded as FAIL, never as a fabricated result.
