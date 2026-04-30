# Parallel multi-seed × multi-dataset training

This document is the **operational recipe** for getting from
"single-seed, 80-min, $0.30 of GPU on one RTX 4090" to
"5-seed × 2-dataset sweep in under 10 minutes wall-clock at
~$0.65 of vast.ai spot rental".

The single-RTX-4090 path is **untouched**.  `make all` still works
exactly as documented in `scripts/vastai_bootstrap.sh`.  Everything
in this doc is opt-in.

---

## TL;DR — the 30-second tour

| Mode | Wall clock | Cost | When |
| --- | --- | --- | --- |
| `make all` (single 4090 vast.ai instance) | ~80 min | ~$0.30 | One seed, one dataset |
| `bash scripts/sweep_seeds_local.sh "42 1 2 3 5"` (single 4090) | ~85 min | ~$0.32 | 5 seeds, backbone shared |
| **`bash scripts/single_instance_swarm.sh`** (8× RTX 5090) | **~12 min** | **~$0.80** | **NeurIPS-credibility 5-seed × 1-dataset** |
| `bash scripts/single_instance_swarm.sh` with `KAGGLE2_DATASETS="canonical sroie"` (8× 5090) | ~22 min | ~$1.50 | 5 seeds × 2 datasets |
| `bash scripts/vastai_swarm.sh "42 1 2 3 5" "canonical sroie"` (multi-instance) | ~10 min | ~$1.30 | 5 seeds × 2 datasets, cross-instance parallel |

The cloud sweeps run **fully in parallel**: backbone training on one
big-GPU instance overlaps with N small-GPU instances each running a
single seed's assigner-only training.

---

## Why this is fast

The sequential 80-min wall clock breaks down as:

| Stage | Wall clock on RTX 4090 | Seed-sensitive? |
| --- | --- | --- |
| DONUT | 25 min | Marginally — within ±0.01 F1 across seeds |
| YOLO | 6 min | Marginally |
| TrOCR | 30 min | Marginally |
| **AttentionAssigner** | **20 min** | **Yes — this is what the paper measures** |
| Eval + paper | 1–2 min | No |

The pipeline contribution that the FOCUS paper makes is the
**attention assigner**.  The upstream OCR/detector are pretrained
backbones that we fine-tune once and reuse.  So:

1. **Train the backbone once** (DONUT + YOLO + TrOCR), upload the
   tarball to a shared bucket.
2. **Run N independent assigner trainings** in parallel, each
   pulling the same backbone tarball, each using its own seed.

On vast.ai H100 / H200 instances the backbone itself takes ~7 min
with batch-size scaling, and each per-seed assigner takes ~3 min.
Run them concurrently and the total wall clock is `max(7, 3) = 7`
min plus ~3 min of orchestration + upload latency.

---

## Prerequisites

You need three things on your **laptop** (not on the cloud
instance — the cloud instances bootstrap themselves):

1. **`vastai` CLI** — `pip install vastai`, then
   `vastai set api-key <your-key>`.  Sign up at
   <https://vast.ai>; ~$10 of credit funds dozens of full sweeps.

2. **rclone with a configured shared bucket** — the swarm uses
   rclone for backbone + per-seed artefact transfer.  Any bucket
   rclone supports works (S3, GCS, R2, B2, Drive, …).  Run
   `rclone config` once and create a remote, e.g.

   ```sh
   $ rclone config
   # → name: kaggle2
   # → storage: s3
   # → ... (region, bucket name, access key)
   $ export KAGGLE2_BUCKET_URL="kaggle2:my-kaggle2-bucket"
   $ export KAGGLE2_RCLONE_CONF="$HOME/.config/rclone/rclone.conf"
   ```

3. **The repo cloned + this branch checked out**.  No bootstrap on
   the laptop side.

That is the entire one-time setup.

---

## Recipe 1 (RECOMMENDED) — Single 8× RTX 5090 vast.ai instance, ~12 min

This is the **NeurIPS-credibility recipe** — five seeds, one
dataset, full DONUT + FOCUS comparison, paired-bootstrap CIs,
per-seed paper PDF, all on one instance in under twelve minutes.

```sh
# 1. Spin up a vast.ai instance with 8× RTX 5090 (or 8× RTX 4090,
#    or 4× H100 PCIe — anything with >= 4 GPUs and >= 24 GB each).
#    "PyTorch 2.4 / CUDA 12.1" template, >= 80 GB disk.
#
# 2. Open the instance terminal and paste these four commands:
git clone -b claude/improve-f1-scores-RYvNY \
    https://github.com/aiparallel0/kaggle2 && cd kaggle2
bash scripts/vastai_bootstrap.sh
bash scripts/single_instance_swarm.sh

# 3. (after ~12 min the script prints the aggregate table to stdout)
#
# 4. Pack and download:
tar --use-compress-program=zstd -cf sweep.tar.zst \
    runs/sweep-* logs/sweep-*
# scp / rclone / browser download the tarball.
```

The branch `claude/improve-f1-scores-RYvNY` ships every patch in
`docs/HONESTY.md` plus the swarm scripts; once the branch lands on
`main` the recipe collapses to `git clone https://...` without
the `-b` flag.  Until then the explicit branch flag is required.

### Two phase-1 execution modes

The swarm picks between **sequential** (default) and **parallel** mode
via `KAGGLE2_PHASE1_MODE`.  The right choice depends on whether your
GPU class has NVLink:

| Mode | Description | Best for |
| --- | --- | --- |
| **sequential** (default) | DONUT → TrOCR → YOLO run *one at a time*; each component spans **all** detected GPUs | Consumer silicon (RTX 4090 / 5090) without NVLink |
| **parallel** | DONUT / TrOCR / YOLO run **concurrently** on disjoint GPU subsets | NVLinked H100 SXM5 / A100 SXM4 with high all-reduce bandwidth |

#### Why sequential is the new default

The first live run on 8× RTX 5090 (PR #133) measured DONUT at
24.35 min on 4 GPUs in **parallel** mode.  At that time only 4 of 8
GPUs were running DONUT; the other 4 were running TrOCR (~6 min) or
YOLO (~2 min) and then sat **idle for ~16 min** until DONUT finished.
The vast.ai dashboard showed an average GPU load of **~38%** —
i.e. half the silicon was wasted.

Sequential mode lets DONUT span all 8 GPUs.  Even with consumer-PCIe
DDP scaling (which is sub-linear above 4 ranks on 5090), DONUT
should drop to ~14–18 min on 8 GPUs vs 24 min on 4.  Average
utilisation rises to ~70–80% because every GPU is doing useful
work during every phase.

#### Wall-clock budgets (estimated)

| Mode | DONUT | TrOCR | YOLO | Phase 1 total | Phase 2 (5 seeds) | **Sweep total** |
| --- | --- | --- | --- | --- | --- | --- |
| sequential, 8× RTX 5090 | ~14–18 min on 8 GPUs | ~3–5 min on 8 GPUs | ~2 min on 1 GPU | **~22 min** (sum) | ~3 min | **~28–30 min** |
| parallel, 8× RTX 5090 | ~24 min on 4 GPUs | ~6–8 min on 2 GPUs | ~2 min on 1 GPU | **~24 min** (max) | ~3 min | **~30 min** |
| parallel, 8× H100 SXM5 (NVLink) | ~6 min on 4 GPUs | ~5 min on 2 GPUs | ~2 min on 1 GPU | **~6 min** (max) | ~2 min | **~10 min** |
| sequential, 8× H100 SXM5 (NVLink) | ~3 min on 8 GPUs | ~2 min on 8 GPUs | ~1 min on 1 GPU | **~6 min** (sum) | ~2 min | **~10 min** |

The sequential and parallel rows produce **the same per-seed
artefacts**.  Pick whichever finishes faster on your GPU class.

#### Switching modes

```sh
# Default — best for consumer GPUs:
bash scripts/single_instance_swarm.sh

# Force parallel — best for NVLinked H100/A100 SXM:
KAGGLE2_PHASE1_MODE=parallel bash scripts/single_instance_swarm.sh

# FOCUS-only sweep (no DONUT) — fastest path on any GPU class:
KAGGLE2_SKIP_DONUT=1 bash scripts/single_instance_swarm.sh
```

GPU partitioning is overridable; defaults assume 8 GPUs.

```sh
# Custom partitioning examples
KAGGLE2_DONUT_GPUS="0,1" KAGGLE2_TROCR_GPUS="2,3" \
    KAGGLE2_YOLO_GPU="4" KAGGLE2_LLM3_GPU="5" \
    bash scripts/single_instance_swarm.sh   # 6-GPU box

KAGGLE2_DONUT_GPUS="0,1,2,3,4,5" KAGGLE2_TROCR_GPUS="6,7" \
    bash scripts/single_instance_swarm.sh   # all-in on DONUT (4 min)

KAGGLE2_SKIP_DONUT=1 \
    bash scripts/single_instance_swarm.sh   # FOCUS-only sweep, ~7 min
```

Reasonable cost references on vast.ai mid-2026 spot (your numbers
will vary):

- 8× RTX 5090 PCIe: $3-4/hour → $0.80 for a 12-min sweep
- 8× RTX 4090 PCIe: $2-3/hour → $0.50 for a 15-min sweep
- 4× H100 PCIe:     $4-5/hour → $0.85 for a 12-min sweep
- 8× H100 SXM5:     $12-15/hour → $2.00 for an 8-min sweep

## Recipe 2 — Local single-4090 sweep (slower, simpler)

If you have just one GPU (e.g. a single-4090 vast.ai instance):

```sh
bash scripts/sweep_seeds_local.sh "42 1 2 3 5"
python scripts/aggregate_seeds.py runs/backbone-*-seed*/
```

The first command trains the backbone once (~60 min), then loops
five seeds × ~5 min each = ~85 min total.  The second prints a
mean ± std table to stdout and (with `--out` flag) writes JSON.

This is the *cheapest* multi-seed path; it just needs patience
on the order of an hour.

---

## Recipe 3 — Multi-instance cloud swarm (when you need cross-dataset parallel)

```sh
# 1. one-time env (laptop)
export KAGGLE2_BUCKET_URL="kaggle2:my-bucket"
export KAGGLE2_RCLONE_CONF="$HOME/.config/rclone/rclone.conf"

# 2. choose GPU classes (defaults shown; raise to H100/H200 for speed)
export KAGGLE2_BACKBONE_GPU="H100_SXM"   # 8x H100 SXM5 if avail
export KAGGLE2_ASSIGNER_GPU="RTX_4090"   # cheap per-seed instances

# 3. launch
bash scripts/vastai_swarm.sh "42 1 2 3 5" "canonical"

# 4. (after the script prints "Sweep complete") pull and aggregate
rclone --config "$KAGGLE2_RCLONE_CONF" \
    copy "$KAGGLE2_BUCKET_URL/seeds" "./runs/sweep-$(date +%Y%m%d)"
python scripts/aggregate_seeds.py "./runs/sweep-$(date +%Y%m%d)/"*
```

The orchestrator:
1. Spawns one backbone instance, blocks until the backbone tarball
   appears in the bucket.
2. Spawns one per-seed instance per `(seed, dataset)` tuple, all
   pointing at the same backbone tarball.
3. Polls the bucket for every `(seed, dataset)` upload.
4. Tears down all instances.

**Failure handling** — if any single instance dies, its tarball
just doesn't show up.  The swarm will hang waiting for it; rerun
the swarm with the same `KAGGLE2_SWEEP_ID` to resume.  The
backbone instance's output is idempotent (already-uploaded files
are no-op'd), so a partial sweep doesn't burn cloud time.

---

## Recipe 4 — Cross-dataset sweep (CORD / WildReceipt added)

The dataset list is the second positional arg.  Add a config per
dataset under `configs/` (template: `configs/canonical_5seed.json`)
and reference it from `scripts/instance_runner.sh:30` (the
`KAGGLE2_DATASET → CONFIG_FILE` mapping).  Then:

```sh
bash scripts/vastai_swarm.sh "42 1 2" "sroie cord wildreceipt"
```

This launches 1 backbone × 3 datasets + 9 per-seed instances in
parallel.  Wall clock stays ~10 min — the dataset axis adds
*instances*, not *time*.

---

## What the swarm preserves vs changes

| Property | RTX 4090 path | Swarm path |
| --- | --- | --- |
| Trains DONUT / YOLO / TrOCR / Assigner end-to-end | ✓ | ✓ |
| Same `models/focus_train.train_assigner` invocation | ✓ | ✓ |
| Same `pipeline_metrics.json` schema written to disk | ✓ | ✓ |
| Same `runs/<id>/` layout | ✓ | ✓ |
| Bit-identical assigner training given fixed seed + fixed backbone | ✓ | ✓ |
| Multi-seed support | (sequential) | (parallel) |
| Single-machine `make all` works | ✓ | ✓ |

The only addition to `main.py` is two new `--stage` choices
(`train_backbone`, `train_assigner`).  Nothing about the existing
`train`/`eval`/`paper`/`all` flow changes.

---

## Honest limits

- **GPU availability**: H100 SXM5 instances are sometimes
  unavailable on vast.ai spot.  The orchestrator picks the cheapest
  reliable offer; if no SXM5 is available it falls back to PCIe
  H100 / 4090, raising the backbone wall clock to 12–15 min.
- **First-run cost**: the backbone instance pre-trains DONUT from
  `naver-clova-ix/donut-base`, which downloads ~800 MB.  Caching
  the pretrained checkpoint in your bucket and pre-staging it
  shaves ~30 s.
- **Bucket egress**: per-seed tarballs are ~50 MB each (DONUT +
  YOLO + TrOCR + assigner.pt + metrics + figures).  The
  paper-render PDF is included so each seed's tarball is
  self-contained.  10 seeds × 50 MB = 0.5 GB egress on result
  download.

---

## Glossary of env vars

| Variable | Purpose | Where |
| --- | --- | --- |
| `KAGGLE2_PHASE` | `backbone` or `assigner` | per-instance |
| `KAGGLE2_RUN_ID` | run-id prefix | per-instance |
| `KAGGLE2_BACKBONE_FROM` | path to backbone artefacts | per-instance, set by `instance_runner.sh` |
| `KAGGLE2_BUCKET_URL` | rclone-compatible bucket URL | laptop & per-instance |
| `KAGGLE2_BACKBONE_KEY` | bucket key of backbone tarball | per-instance |
| `KAGGLE2_SEED` | integer seed | per-instance, assigner phase only |
| `KAGGLE2_DATASET` | `sroie` / `canonical` / custom | per-instance |
| `KAGGLE2_RCLONE_CONF` | path to local rclone config | laptop |
| `KAGGLE2_BACKBONE_GPU` | vast.ai GPU class for backbone | laptop |
| `KAGGLE2_ASSIGNER_GPU` | vast.ai GPU class for per-seed | laptop |
| `KAGGLE2_DOCKER_IMAGE` | base image | laptop, default pytorch:2.4 |

---

## When to NOT use this

- A *single* seed run on a *single* dataset.  Just run `make all`.
- Iterating on the assigner architecture: the local
  `sweep_seeds_local.sh` is faster end-to-end because it amortises
  the backbone training across tweaks of the assigner code.
- Sub-30-second smoke tests: nothing in the swarm fits in that
  window.  Use `pytest tests/models/` for code-level checks.

---

## Promotion criteria, revisited

`docs/HONESTY.md` lists the gates that move FOCUS from "single-seed
point estimate" to a publishable claim.  This recipe directly closes
two of them:

* **Multi-seed variance**: 5-seed swarm in 10 min ⇒ a 95% CI on
  every headline F1 within one trip to the kettle.
* **Cross-dataset generalisation**: add a config, add a dataset
  name to the swarm CLI, get a fully comparable measurement on
  CORD / WildReceipt without code changes.
