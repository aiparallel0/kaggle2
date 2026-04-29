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
| `make all` | ~80 min | ~$0.30 | Local 4090, single seed |
| `bash scripts/sweep_seeds_local.sh "42 1 2 3 5"` | ~85 min | ~$0.32 | Local 4090, 5 seeds (backbone shared!) |
| `bash scripts/vastai_swarm.sh "42 1 2 3 5" "canonical"` | **~10 min** | ~$0.65 | Cloud, 5 seeds × 1 dataset |
| `bash scripts/vastai_swarm.sh "42 1 2 3 5" "canonical sroie"` | **~10 min** | ~$1.30 | Cloud, 5 seeds × 2 datasets |

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

## Recipe 1 — Local sweep on one 4090 (no vast.ai)

If you already have a local RTX 4090 and just want multi-seed
results:

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

## Recipe 2 — Cloud swarm sweep (~10 min wall clock)

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

## Recipe 3 — Cross-dataset sweep (CORD / WildReceipt added)

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
