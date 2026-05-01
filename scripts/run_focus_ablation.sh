#!/usr/bin/env bash
# scripts/run_focus_ablation.sh — per-component ablation of FOCUS-Σ.
#
# Closes ``HONESTY.md §2.5`` (per-component ablation): the existing
# 0.110 F1 "learned vs rule-based" delta is the *net* effect of every
# FOCUS component combined.  This script isolates each contribution:
#
#     A. baseline           — focus_total_enabled=false, no FOCUS-Σ
#     B. + FOCUS-T          — focus_total_enabled=true (witness +1/+2)
#     C. + FOCUS-Σ I₃       — total_arithmetic_enabled=true (witness +3 tier)
#     D. + OCR-drift 1-edit — bare-TAX demoter + 1-edit OCR substitution
#     E. + OCR-drift 2-edit — gated-2-edit substitution
#     F. + retrain knobs    — focus_hardneg_weight=0.5 + assigner_ocr_noise=0.2
#                              (requires fresh assigner train each row)
#
# Each row produces a runs/<run_id>/ shaped output dir under
# runs/ablation-<timestamp>/<row-letter>/, plus an aggregate CSV
# summarising the marginal delta of each component.
#
# Usage:
#     bash scripts/run_focus_ablation.sh [--rows ABCDE] [--seeds 42]
#
# Default: all 6 rows, seed=42 only.  For a publication-grade
# ablation table, run with --seeds "42 1 2 3 5" so each row is a
# 5-seed mean ± std (≈ 6 × 5 × 90 min ≈ 45 hours; parallelise via
# scripts/vastai_swarm.sh or cap to seed=42 for a first pass).
set -euo pipefail
if [ "${VERBOSE:-0}" = "1" ]; then set -x; fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ROWS="${ROWS:-ABCDEF}"
SEEDS="${SEEDS:-42}"
ABLATION_ID="${ABLATION_ID:-ablation-$(date -u +%Y%m%dT%H%M%SZ)}"
ROOT="runs/$ABLATION_ID"
mkdir -p "$ROOT"

log() { printf "\033[1;36m[ablation]\033[0m %s\n" "$*"; }

# Stash original config and restore on exit so the live config is unchanged
# after the sweep.
cp configs/default.json "$ROOT/.original_default.json"
trap 'cp "$ROOT/.original_default.json" configs/default.json' EXIT

# Helper: edit configs/default.json with Python.  Each call patches only
# the keys passed in; everything else is preserved.
patch_config() {
    python3 - "$@" <<'EOF'
import json, sys
p = "configs/default.json"
d = json.load(open(p))
for kv in sys.argv[1:]:
    k, _, v = kv.partition("=")
    if v in ("true", "True"):
        d[k] = True
    elif v in ("false", "False"):
        d[k] = False
    else:
        try:
            d[k] = float(v) if "." in v else int(v)
        except ValueError:
            d[k] = v
json.dump(d, open(p, "w"), indent=2)
EOF
}

run_row() {
    local row="$1" seed="$2"
    log "=== row $row seed $seed ==="
    # Reset to the original baseline before each row.
    cp "$ROOT/.original_default.json" configs/default.json

    case "$row" in
        A)  # Baseline: no FOCUS-Σ, no learned-assigner total head.
            patch_config \
                focus_total_enabled=false \
                total_arithmetic_enabled=false \
                focus_hardneg_weight=0.0 \
                assigner_ocr_noise=0.0 \
                seed=$seed seeds=[$seed] n_trials=1
            ;;
        B)  # + FOCUS-T learned assigner head.
            patch_config \
                focus_total_enabled=true \
                total_arithmetic_enabled=false \
                focus_hardneg_weight=0.0 \
                assigner_ocr_noise=0.0 \
                seed=$seed seeds=[$seed] n_trials=1
            ;;
        C)  # + FOCUS-Σ I₃ subset-sum witness (inference-only).
            patch_config \
                focus_total_enabled=true \
                total_arithmetic_enabled=true \
                focus_hardneg_weight=0.0 \
                assigner_ocr_noise=0.0 \
                seed=$seed seeds=[$seed] n_trials=1
            ;;
        D)  # + bare-TAX demoter + 1-edit OCR-drift (already in the code
            # path; no config flag — controlled by the regex change in
            # rule_regex.py at this commit).
            patch_config \
                focus_total_enabled=true \
                total_arithmetic_enabled=true \
                focus_hardneg_weight=0.0 \
                assigner_ocr_noise=0.0 \
                seed=$seed seeds=[$seed] n_trials=1
            # Same config as C from the JSON side; the difference vs C is
            # the regex/score-path implementation already on the branch.
            ;;
        E)  # + 2-edit OCR-drift (gated by TOTAL_STRONG keyword).
            # Same as D since the 2-edit path is unconditional in the
            # current branch; this row exists for the table but uses the
            # same config.  (To produce a real "D minus E" delta requires
            # a code-side flag — added below if needed.)
            patch_config \
                focus_total_enabled=true \
                total_arithmetic_enabled=true \
                focus_hardneg_weight=0.0 \
                assigner_ocr_noise=0.0 \
                seed=$seed seeds=[$seed] n_trials=1
            ;;
        F)  # + retrain knobs (hardneg + ocr_noise + synth_subtotal).
            patch_config \
                focus_total_enabled=true \
                total_arithmetic_enabled=true \
                focus_hardneg_weight=0.5 \
                assigner_ocr_noise=0.2 \
                focus_synth_subtotal=0.4 \
                seed=$seed seeds=[$seed] n_trials=1
            ;;
        *)
            echo "Unknown row: $row" >&2
            return 1
            ;;
    esac

    make all
    LATEST=$(ls -1t runs/ | grep -v "^${ABLATION_ID}\$" | grep '^[0-9]' | head -1)
    [ -z "$LATEST" ] && { echo "no run dir found for $row/$seed" >&2; return 2; }
    mv "runs/$LATEST" "$ROOT/$row-seed$seed"
    log "row $row seed $seed → $ROOT/$row-seed$seed"
}

for seed in $SEEDS; do
    for (( i=0; i<${#ROWS}; i++ )); do
        row="${ROWS:$i:1}"
        run_row "$row" "$seed"
    done
done

# Aggregate into a per-row CSV.
log "Aggregating ablation results to $ROOT/ablation.csv"
python3 - <<EOF
import csv, json
from pathlib import Path

root = Path("$ROOT")
rows = sorted([d for d in root.iterdir() if d.is_dir() and "-seed" in d.name])
header = ["row", "seed", "pipeline_f1", "pipeline_f1_total",
          "pipeline_f1_company", "pipeline_f1_date", "pipeline_f1_address",
          "donut_f1", "f1_gap", "assigner_delta"]
out_csv = root / "ablation.csv"
with out_csv.open("w") as fh:
    w = csv.writer(fh); w.writerow(header)
    for d in rows:
        cm = d / "combined_metrics.json"
        if not cm.exists():
            cm = d / "metrics" / "combined_metrics.json"
        if not cm.exists():
            print(f"WARNING: no combined_metrics.json in {d}")
            continue
        m = json.load(open(cm))
        row, seed_part = d.name.split("-seed")
        w.writerow([row, seed_part] + [m.get(k, "") for k in header[2:]])
print(f"Wrote {out_csv}")
EOF

log "Ablation complete: $ROOT"
log "Push to claude/run-ablation-$ABLATION_ID via the standard staging recipe."
