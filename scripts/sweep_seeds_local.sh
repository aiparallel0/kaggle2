#!/usr/bin/env bash
# scripts/sweep_seeds_local.sh
# Local single-GPU multi-seed sweep with backbone caching.
#
# Trains DONUT + YOLO + TrOCR ONCE, then loops over seeds running only
# the AttentionAssigner + eval stages.  On an RTX 4090 this turns a
# 5-seed sweep from ~7 hours (5 × 80 min full pipeline) into
# ~85 min (one 80-min backbone + 5 × 1-min assigner sweeps).
#
# Usage (from repo root):
#     bash scripts/sweep_seeds_local.sh "42 1 2 3 5"
#     bash scripts/sweep_seeds_local.sh                  # defaults to 5 seeds
#
# Output: ./runs/<id>-seed<N>/ for each seed N, with a shared backbone
#         materialised by symlink so disk usage stays bounded.
#         The combined paper-render and aggregator hooks are skipped
#         here — see scripts/aggregate_seeds.py for the post-step.

set -euo pipefail
SEEDS="${1:-42 1 2 3 5}"
CONFIG="${2:-configs/default.json}"

log() { printf "\033[1;36m[sweep]\033[0m %s\n" "$*"; }

# Phase 1: train shared backbone (DONUT + YOLO + TrOCR) once.
BACKBONE_RUN_ID="backbone-$(date -u +%Y%m%dT%H%M%SZ)"
BACKBONE_DIR="runs/${BACKBONE_RUN_ID}"
log "Phase 1: training shared backbone at ${BACKBONE_DIR}"
KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
    python main.py --stage train_backbone --config "${CONFIG}"

# Phase 2: per-seed assigner + eval, all reusing the backbone.
for SEED in ${SEEDS}; do
    SEED_RUN_ID="${BACKBONE_RUN_ID}-seed${SEED}"
    SEED_DIR="runs/${SEED_RUN_ID}"
    log "Phase 2: seed=${SEED} → ${SEED_DIR}"
    KAGGLE2_RUN_ID="${SEED_RUN_ID}" \
    KAGGLE2_BACKBONE_FROM="${BACKBONE_DIR}" \
        python main.py --stage train_assigner --config "${CONFIG}" \
        --seeds "${SEED}"
    KAGGLE2_RUN_ID="${SEED_RUN_ID}" \
        python main.py --stage eval --config "${CONFIG}" --seeds "${SEED}"
done

log "Sweep complete.  Per-seed runs:"
for SEED in ${SEEDS}; do
    log "  runs/${BACKBONE_RUN_ID}-seed${SEED}/"
done
log "Aggregate with:  python scripts/aggregate_seeds.py runs/${BACKBONE_RUN_ID}*"
