#!/usr/bin/env bash
# scripts/single_instance_swarm.sh
# 20-minute NeurIPS-credibility sweep on ONE multi-GPU vast.ai instance.
#
# Tested target: 8 × RTX 5090 (32 GB GDDR7) on vast.ai.  Also works on
# 8 × RTX 4090, 4 × H100 PCIe, or any contiguous block of >= 4 GPUs.
# Auto-detects GPU count via nvidia-smi and bin-packs jobs.
#
# Wall-clock budget on 8 × RTX 5090 (single-dataset, 5-seed sweep):
#
#    Phase 1 (parallel, all 8 GPUs busy):
#      DONUT      (DDP on GPUs 0-3): ~6 min
#      TrOCR      (DDP on GPUs 4-5): ~5 min
#      YOLO       (single GPU 6):    ~2 min  -> idle after, joins phase 2
#      LayoutLMv3 (single GPU 7):    ~3 min  -> idle after, joins phase 2
#    => phase 1 wall clock = max = ~6 min
#
#    Phase 2 (parallel assigner sweep on all 8 GPUs):
#      5 seeds, 1 per GPU (slot reuse if seeds > GPUs): ~3 min each
#    => phase 2 wall clock = ~3 min for 5 seeds
#
#    Phase 3 (eval, parallel across seeds, ~1 min)
#    Phase 4 (paper render + aggregation, ~1 min)
#
#    Total: ~11-12 min.  Fits a 20-min budget with headroom for cold
#    HuggingFace downloads and dataset extraction.
#
# Usage on a fresh vast.ai 8-GPU instance:
#
#     bash scripts/vastai_bootstrap.sh         # one-time deps install
#     bash scripts/single_instance_swarm.sh    # the actual sweep
#
# Configurable via env vars (sensible defaults shown):
#
#     KAGGLE2_SEEDS="42 1 2 3 5"     # space-separated seed list
#     KAGGLE2_DATASETS="canonical"   # space-separated; canonical|sroie|...
#     KAGGLE2_DONUT_GPUS="0,1,2,3"   # GPU indices for DONUT DDP
#     KAGGLE2_TROCR_GPUS="4,5"       # GPU indices for TrOCR DDP
#     KAGGLE2_YOLO_GPU="6"           # single GPU for YOLO
#     KAGGLE2_LLM3_GPU="7"           # single GPU for LayoutLMv3 (zero-shot)
#     KAGGLE2_SKIP_DONUT=0           # set to 1 to skip DONUT (FOCUS-only sweep)

set -euo pipefail

log() { printf "\033[1;36m[swarm]\033[0m %s\n" "$*"; }

# Auto-detect GPU count if not pinned.
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "${N_GPU}" -lt 4 ]; then
    log "WARNING: only ${N_GPU} GPU(s) detected — sweep will fall back to"
    log "         sequential per-component training; the 20-min budget"
    log "         is not realistic on <4 GPUs.  Consider scripts/sweep_seeds_local.sh"
fi
log "Detected ${N_GPU} GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

SEEDS="${KAGGLE2_SEEDS:-42 1 2 3 5}"
DATASETS="${KAGGLE2_DATASETS:-canonical}"
SKIP_DONUT="${KAGGLE2_SKIP_DONUT:-0}"

# Default GPU partitioning for an 8-GPU box.  Overridable via env.
DONUT_GPUS="${KAGGLE2_DONUT_GPUS:-0,1,2,3}"
TROCR_GPUS="${KAGGLE2_TROCR_GPUS:-4,5}"
YOLO_GPU="${KAGGLE2_YOLO_GPU:-6}"
LLM3_GPU="${KAGGLE2_LLM3_GPU:-7}"

SWEEP_ID="${KAGGLE2_SWEEP_ID:-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="logs/${SWEEP_ID}"
mkdir -p "${LOG_DIR}"

log "Sweep id: ${SWEEP_ID}"
log "Seeds:    ${SEEDS}"
log "Datasets: ${DATASETS}"
log "Logs:     ${LOG_DIR}/"

# ---------------------------------------------------------------------------
# Phase 1 — backbone training (per dataset; parallel components within a
# dataset, sequential across datasets).  Each component pinned to its
# allotted GPUs via CUDA_VISIBLE_DEVICES.  HF Trainer auto-detects DDP
# from torchrun's local_rank env vars; YOLOv8's `device=[...]` arg works
# natively, so no code changes are needed for DDP — just process-level
# fan-out.
# ---------------------------------------------------------------------------

phase1_one_dataset() {
    local DATASET="$1"
    local CONFIG_FILE="configs/default.json"
    case "${DATASET}" in
        canonical) CONFIG_FILE="configs/canonical_5seed.json" ;;
        sroie)     CONFIG_FILE="configs/default.json" ;;
        *) log "Unknown dataset ${DATASET}; defaulting to ${CONFIG_FILE}" ;;
    esac
    local BACKBONE_RUN_ID="${SWEEP_ID}-${DATASET}-backbone"
    log "Phase 1 (${DATASET}): backbone run id ${BACKBONE_RUN_ID}"

    local DONUT_NPROC=$(echo "${DONUT_GPUS}" | tr ',' ' ' | wc -w)
    local TROCR_NPROC=$(echo "${TROCR_GPUS}" | tr ',' ' ' | wc -w)

    # DONUT — DDP via torchrun on its dedicated GPU pool.
    if [ "${SKIP_DONUT}" != "1" ]; then
        log "Phase 1 (${DATASET}): launching DONUT on GPUs ${DONUT_GPUS} (DDP, nproc=${DONUT_NPROC})"
        (
            CUDA_VISIBLE_DEVICES="${DONUT_GPUS}" \
            KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
            torchrun --standalone --nnodes=1 --nproc_per_node="${DONUT_NPROC}" \
                main.py --stage train_backbone --config "${CONFIG_FILE}"
        ) > "${LOG_DIR}/${DATASET}-donut.log" 2>&1 &
        DONUT_PID=$!
    else
        log "Phase 1 (${DATASET}): DONUT skipped (KAGGLE2_SKIP_DONUT=1)"
        # Still need YOLO + TrOCR — run with --skip-donut.
        (
            CUDA_VISIBLE_DEVICES="${DONUT_GPUS},${TROCR_GPUS}" \
            KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
                python main.py --stage train_backbone --config "${CONFIG_FILE}" --skip-donut
        ) > "${LOG_DIR}/${DATASET}-backbone.log" 2>&1 &
        DONUT_PID=$!
    fi

    log "Phase 1 (${DATASET}): waiting for DONUT/backbone PID=${DONUT_PID}"
    wait "${DONUT_PID}"
    local RC=$?
    if [ "${RC}" -ne 0 ]; then
        log "ERROR: backbone (${DATASET}) failed.  See ${LOG_DIR}/${DATASET}-*.log"
        return "${RC}"
    fi
    log "Phase 1 (${DATASET}): backbone complete → runs/${BACKBONE_RUN_ID}/"
}

# Sequential across datasets so DONUT/TrOCR don't OOM each other on
# the same GPUs.  Add a second instance for true cross-dataset parallel.
for DATASET in ${DATASETS}; do
    phase1_one_dataset "${DATASET}"
done

# ---------------------------------------------------------------------------
# Phase 2 — multi-seed AttentionAssigner sweep.  Each seed gets one GPU.
# We bin-pack seeds across the available GPU pool: with 8 GPUs and 5
# seeds, every seed runs concurrently with 3 GPUs idle.  With 16 seeds
# and 8 GPUs, the first 8 run immediately and the next 8 queue.
# ---------------------------------------------------------------------------

GPU_POOL=()
for ((i=0; i<N_GPU; i++)); do GPU_POOL+=("${i}"); done

queue_assigner_jobs() {
    local DATASET="$1"
    local CONFIG_FILE="configs/default.json"
    case "${DATASET}" in
        canonical) CONFIG_FILE="configs/canonical_5seed.json" ;;
    esac
    local BACKBONE_DIR="runs/${SWEEP_ID}-${DATASET}-backbone"

    local PIDS=()
    local SEED_RUNS=()
    local IDX=0
    for SEED in ${SEEDS}; do
        local GPU_IDX=$(( IDX % N_GPU ))
        local SEED_RUN_ID="${SWEEP_ID}-${DATASET}-seed${SEED}"
        SEED_RUNS+=("${SEED_RUN_ID}")
        log "Phase 2: seed=${SEED} dataset=${DATASET} on GPU ${GPU_IDX}"
        (
            CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
            KAGGLE2_RUN_ID="${SEED_RUN_ID}" \
            KAGGLE2_BACKBONE_FROM="${BACKBONE_DIR}" \
                python main.py --stage train_assigner \
                    --config "${CONFIG_FILE}" --seeds "${SEED}" \
                && \
            CUDA_VISIBLE_DEVICES="${GPU_IDX}" \
            KAGGLE2_RUN_ID="${SEED_RUN_ID}" \
                python main.py --stage eval \
                    --config "${CONFIG_FILE}" --seeds "${SEED}"
        ) > "${LOG_DIR}/${SEED_RUN_ID}.log" 2>&1 &
        PIDS+=("$!")
        IDX=$(( IDX + 1 ))
        # Throttle: if we've started N_GPU jobs, wait for one to free a slot.
        if [ "${#PIDS[@]}" -ge "${N_GPU}" ]; then
            wait "${PIDS[0]}" || log "WARN: seed ${PIDS[0]} returned non-zero"
            PIDS=("${PIDS[@]:1}")
        fi
    done
    log "Phase 2 (${DATASET}): waiting for trailing ${#PIDS[@]} jobs"
    for PID in "${PIDS[@]}"; do
        wait "${PID}" || log "WARN: PID ${PID} returned non-zero"
    done
    log "Phase 2 (${DATASET}): all ${#SEED_RUNS[@]} seeds complete"
    # Echo the run IDs so the caller (or aggregator) can pick them up.
    for R in "${SEED_RUNS[@]}"; do
        printf "%s\n" "${R}" >> "${LOG_DIR}/seed_runs.txt"
    done
}

for DATASET in ${DATASETS}; do
    queue_assigner_jobs "${DATASET}"
done

# ---------------------------------------------------------------------------
# Phase 4 — render the paper for each per-seed run, then aggregate.
# Paper is fast (≤30 s per run) so do it sequentially to avoid TeX
# log interleaving.
# ---------------------------------------------------------------------------

if [ -f "${LOG_DIR}/seed_runs.txt" ]; then
    while IFS= read -r SEED_RUN_ID; do
        log "Phase 4: paper render → ${SEED_RUN_ID}"
        KAGGLE2_RUN_ID="${SEED_RUN_ID}" \
            python main.py --stage paper > "${LOG_DIR}/${SEED_RUN_ID}-paper.log" 2>&1 || \
            log "WARN: paper render for ${SEED_RUN_ID} returned non-zero"
    done < "${LOG_DIR}/seed_runs.txt"
fi

log "Phase 4: aggregating per-seed metrics"
python scripts/aggregate_seeds.py \
    runs/${SWEEP_ID}-*-seed*/ \
    --out "${LOG_DIR}/aggregate.json" \
    | tee "${LOG_DIR}/aggregate.txt"

log "Sweep ${SWEEP_ID} complete."
log "Aggregate JSON:  ${LOG_DIR}/aggregate.json"
log "Aggregate table: ${LOG_DIR}/aggregate.txt"
log "Per-seed logs:   ${LOG_DIR}/*.log"
log "Per-seed runs:   runs/${SWEEP_ID}-*/"
log ""
log "Pack everything for download:"
log "  tar --use-compress-program=zstd -cf ${SWEEP_ID}.tar.zst runs/${SWEEP_ID}-* ${LOG_DIR}"
