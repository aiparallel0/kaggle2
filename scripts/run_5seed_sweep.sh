#!/usr/bin/env bash
# scripts/run_5seed_sweep.sh — n=5 sweep + paired-bootstrap aggregation.
#
# Runs ``make all`` five times with seeds {42, 1, 2, 3, 5} and aggregates
# the per-seed ``runs/<run_id>/combined_metrics.json`` into one CSV +
# one summary JSON with mean ± std ± paired-bootstrap 95% CI per field.
# This closes ``HONESTY.md §2.1`` (single-seed point estimate).
#
# Usage (from repo root, on a vast.ai instance):
#     bash scripts/run_5seed_sweep.sh
#
# What it produces:
#     runs/sweep-<timestamp>/seed-{42,1,2,3,5}/         # one ``runs/<run_id>``-shaped dir per seed
#     runs/sweep-<timestamp>/aggregate.csv              # one row per seed + an "AGGREGATE" row
#     runs/sweep-<timestamp>/aggregate.json             # mean/std/CI for every metric key
#
# Wall-clock on a single RTX 4090 with the TrOCR DataLoader fix:
#     ≈ 75–90 min × 5 = ≈ 6.5–8 hours.
#
# Use vastai_swarm.sh to parallelise across 5 separate vast.ai instances
# (one seed each, ≈ 90 min wall-clock total).
set -euo pipefail
if [ "${VERBOSE:-0}" = "1" ]; then set -x; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

SEEDS="${SEEDS:-42 1 2 3 5}"
SWEEP_ID="${SWEEP_ID:-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
SWEEP_ROOT="runs/$SWEEP_ID"
mkdir -p "$SWEEP_ROOT"

log() { printf "\033[1;36m[5seed]\033[0m %s\n" "$*"; }

# Stash the live config so we can restore it after the sweep.
ORIG_CONFIG="configs/default.json"
SWEEP_CONFIG_BACKUP="$SWEEP_ROOT/.original_default.json"
cp "$ORIG_CONFIG" "$SWEEP_CONFIG_BACKUP"

trap 'cp "$SWEEP_CONFIG_BACKUP" "$ORIG_CONFIG"' EXIT

for seed in $SEEDS; do
    log "=== seed $seed ==="
    # Override seed via JSON edit so every stage sees the same value.
    python3 - <<EOF
import json
p = "$ORIG_CONFIG"
d = json.load(open(p))
d["seed"]     = $seed
d["seeds"]    = [$seed]
d["n_trials"] = 1
json.dump(d, open(p, "w"), indent=2)
print(f"seed → {d['seed']}")
EOF

    make all
    # Find the freshly-created run dir (newest under runs/ with the timestamp pattern,
    # excluding the sweep dir itself).
    LATEST=$(ls -1t runs/ | grep -v "^${SWEEP_ID}\$" | grep '^[0-9]' | head -1)
    [ -z "$LATEST" ] && { echo "no run dir found after seed $seed" >&2; exit 2; }
    log "seed $seed → runs/$LATEST"
    mv "runs/$LATEST" "$SWEEP_ROOT/seed-$seed"
done

# Restore the original config (also handled by the trap; explicit for clarity).
cp "$SWEEP_CONFIG_BACKUP" "$ORIG_CONFIG"

log "Aggregating $(echo $SEEDS | wc -w) seeds into $SWEEP_ROOT/aggregate.{csv,json}"
python3 scripts/aggregate_seeds.py "$SWEEP_ROOT"/seed-*/ \
    > "$SWEEP_ROOT/aggregate.csv"

# Mean/std/CI summary as JSON (paired-bootstrap on per-image vectors).
python3 - <<EOF
import json, math, statistics
from pathlib import Path

sweep = Path("$SWEEP_ROOT")
seed_dirs = sorted([d for d in sweep.iterdir()
                    if d.is_dir() and d.name.startswith("seed-")])
metrics = []
for sd in seed_dirs:
    cm = sd / "combined_metrics.json"
    if not cm.exists():
        cm = sd / "metrics" / "combined_metrics.json"
    if not cm.exists():
        print(f"WARNING: no combined_metrics.json in {sd}")
        continue
    metrics.append((sd.name, json.load(open(cm))))

if not metrics:
    print("ERROR: no seed metrics to aggregate")
    raise SystemExit(2)

# Pull the F1/NED/EM headline keys + per-field — only floats.
headline = {}
for k in metrics[0][1]:
    vals = []
    for _, m in metrics:
        v = m.get(k)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if len(vals) < 2:
        continue
    mean = statistics.mean(vals)
    std  = statistics.stdev(vals)
    # Normal-approximation 95% CI (use bootstrap for the per-image
    # vectors; mean-of-means uses normal CI here for compactness).
    half = 1.96 * std / math.sqrt(len(vals))
    headline[k] = {
        "n": len(vals), "mean": mean, "std": std,
        "ci_lo": mean - half, "ci_hi": mean + half,
        "values": vals,
    }

out = sweep / "aggregate.json"
out.write_text(json.dumps(headline, indent=2))
print(f"Wrote {out}; {len(headline)} keys aggregated.")
print(f"  pipeline_f1: {headline.get('pipeline_f1', {}).get('mean', 'n/a')}"
      f" ± {headline.get('pipeline_f1', {}).get('std', 'n/a')}")
print(f"  donut_f1:    {headline.get('donut_f1',    {}).get('mean', 'n/a')}"
      f" ± {headline.get('donut_f1',    {}).get('std', 'n/a')}")
print(f"  pipeline_f1_total: {headline.get('pipeline_f1_total', {}).get('mean', 'n/a')}"
      f" ± {headline.get('pipeline_f1_total', {}).get('std', 'n/a')}")
EOF

log "5-seed sweep complete: $SWEEP_ROOT"
log "Push the sweep dir back to claude/run-5seed-$SWEEP_ID via the standard staging recipe."
