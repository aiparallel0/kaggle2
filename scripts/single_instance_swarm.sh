#!/usr/bin/env bash
# scripts/single_instance_swarm.sh
# 20-min NeurIPS-credibility sweep on ONE multi-GPU vast.ai instance.
#
# Tested target: 8 × RTX 5090 (32 GB GDDR7) on vast.ai.  Also works on
# 8 × RTX 4090, 4 × H100 PCIe, or any contiguous block of >= 4 GPUs.
# Auto-detects GPU count via nvidia-smi and bin-packs jobs.
#
# Wall-clock budget on 8 × RTX 5090 (single-dataset, 5-seed sweep):
#
#    Phase 0 — prepare data + split (single process)            : ~30 s
#    Phase 1 — backbone components run as separate processes
#              concurrently, each pinned to a disjoint GPU subset:
#      DONUT      (DDP via torchrun on GPUs 0-3) : ~6 min
#      TrOCR      (DDP via torchrun on GPUs 4-5) : ~5 min  } all
#      YOLO       (single process on GPU 6)      : ~2 min  } phase
#      [LayoutLMv3 zero-shot eval skipped here; runs in eval phase]
#    => phase 1 wall clock = max = ~6 min  (DONUT bottleneck)
#
#    Phase 2 — multi-seed AttentionAssigner sweep
#      5 seeds, 1 per GPU, all 8 GPUs available  : ~3 min
#
#    Phase 3 — eval (parallel across seeds)      : ~1 min
#    Phase 4 — paper render (sequential, ~30 s)  : ~2 min
#
#    Total: ~12 min on 8 × RTX 5090.  Fits a 20-min budget with
#    headroom for cold HuggingFace downloads and dataset extraction.
#
# Usage on a fresh vast.ai 8-GPU instance (PyTorch 2.4 / CUDA 12.1):
#
#     # On vast.ai, after the instance is booted:
#     git clone -b claude/improve-f1-scores-RYvNY \
#         https://github.com/aiparallel0/kaggle2 && cd kaggle2
#     bash scripts/vastai_bootstrap.sh
#     bash scripts/single_instance_swarm.sh
#
# Configurable via env vars (sensible defaults shown):
#
#     KAGGLE2_SEEDS="42 1 2 3 5"      # space-separated seed list
#     KAGGLE2_DATASETS="canonical"    # space-separated; canonical|sroie
#     KAGGLE2_DONUT_GPUS="0,1,2,3"    # GPU indices for DONUT DDP
#     KAGGLE2_TROCR_GPUS="4,5"        # GPU indices for TrOCR DDP
#     KAGGLE2_YOLO_GPU="6"            # single GPU for YOLO
#     KAGGLE2_SKIP_DONUT=0            # 1 = skip DONUT (FOCUS-only sweep)
#
# Compatibility: every existing test, config, and CLI flag continues
# to work.  This script is purely additive on top of `make all`.

set -euo pipefail

log() { printf "\033[1;36m[swarm]\033[0m %s\n" "$*"; }

# Fail loudly when invoked from outside the repo root.
if [ ! -f "main.py" ] || [ ! -d "stages" ]; then
    echo "ERROR: run this from the kaggle2 repo root." >&2
    exit 2
fi

# Auto-detect GPU count if not pinned.
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found.  This script needs CUDA + GPUs." >&2
    exit 2
fi
N_GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "${N_GPU}" -lt 4 ]; then
    log "WARNING: only ${N_GPU} GPU(s) detected — sweep will fall back to"
    log "         sequential per-component training; the 12-min budget is"
    log "         not realistic on <4 GPUs.  Consider:"
    log "         bash scripts/sweep_seeds_local.sh"
fi
log "Detected ${N_GPU} GPU(s): $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"

SEEDS="${KAGGLE2_SEEDS:-42 1 2 3 5}"
DATASETS="${KAGGLE2_DATASETS:-canonical}"
SKIP_DONUT="${KAGGLE2_SKIP_DONUT:-0}"

# Default GPU partitioning for an 8-GPU box.  Overridable via env.
DONUT_GPUS="${KAGGLE2_DONUT_GPUS:-0,1,2,3}"
TROCR_GPUS="${KAGGLE2_TROCR_GPUS:-4,5}"
YOLO_GPU="${KAGGLE2_YOLO_GPU:-6}"

SWEEP_ID="${KAGGLE2_SWEEP_ID:-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"
LOG_DIR="logs/${SWEEP_ID}"
mkdir -p "${LOG_DIR}"

log "Sweep id: ${SWEEP_ID}"
log "Seeds:    ${SEEDS}"
log "Datasets: ${DATASETS}"
log "Logs:     ${LOG_DIR}/"

# Partial config overlays (configs/canonical_5seed.json etc.) ship only
# the keys they override and rely on a downstream consumer to fold them
# onto configs/default.json before load_config is called.  main.py itself
# does not perform that merge — passing a partial config directly fails
# load_config with "missing required keys".  Resolve here: when the named
# dataset has an overlay file, materialise the merged JSON to /tmp and
# return its path; otherwise return configs/default.json directly.
resolve_config() {
    local DATASET="$1"
    local OVERLAY=""
    case "${DATASET}" in
        canonical) OVERLAY="configs/canonical_5seed.json" ;;
        sroie)     echo "configs/default.json"; return ;;
        *)         echo "configs/default.json"; return ;;
    esac
    if [ ! -f "${OVERLAY}" ]; then
        echo "configs/default.json"
        return
    fi
    local MERGED="/tmp/kaggle2-config-${DATASET}.json"
    python -c "
import json, sys
base = json.load(open('configs/default.json'))
base.update(json.load(open('${OVERLAY}')))
json.dump(base, open('${MERGED}', 'w'), indent=2)
" || { echo "ERROR: failed to merge ${OVERLAY} onto configs/default.json" >&2; return 1; }
    echo "${MERGED}"
}

# Stream the relevant log file to stderr on a failed phase so the user
# does not have to grep through logs/ to find the actual error.
dump_log_on_fail() {
    local TAG="$1" PATH_="$2"
    if [ -f "${PATH_}" ]; then
        echo "" >&2
        echo "----- ${TAG} (last 60 lines of $(basename "${PATH_}")) -----" >&2
        tail -n 60 "${PATH_}" >&2
        echo "----- end ${TAG} -----" >&2
        echo "" >&2
    fi
}

# ---------------------------------------------------------------------------
# Phase 1 — backbone components run AS PARALLEL PROCESSES, each pinned to
# its allotted GPUs via CUDA_VISIBLE_DEVICES.  HF Trainer (DONUT, TrOCR)
# auto-detects DDP from torchrun's local_rank env vars; YOLOv8's
# `device=[...]` arg works natively, so no training-code changes are
# needed for DDP — just process-level fan-out.
# ---------------------------------------------------------------------------

phase1_one_dataset() {
    local DATASET="$1"
    local CONFIG_FILE
    CONFIG_FILE="$(resolve_config "${DATASET}")"
    log "Phase 0 (${DATASET}): config = ${CONFIG_FILE}"
    local BACKBONE_RUN_ID="${SWEEP_ID}-${DATASET}-backbone"
    log "Phase 0 (${DATASET}): preparing data + split (${BACKBONE_RUN_ID})"
    if ! KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
        python main.py --stage prepare_data --config "${CONFIG_FILE}" \
        > "${LOG_DIR}/${DATASET}-prepare.log" 2>&1; then
        log "  prepare_data FAILED"
        dump_log_on_fail "${DATASET}-prepare" "${LOG_DIR}/${DATASET}-prepare.log"
        return 1
    fi

    log "Phase 1 (${DATASET}): launching parallel components"
    local DONUT_NPROC TROCR_NPROC
    DONUT_NPROC=$(echo "${DONUT_GPUS}" | tr ',' ' ' | wc -w)
    TROCR_NPROC=$(echo "${TROCR_GPUS}" | tr ',' ' ' | wc -w)

    # ---- DONUT (DDP, GPUs 0-3 by default) --------------------------------
    if [ "${SKIP_DONUT}" != "1" ]; then
        log "  DONUT  (DDP nproc=${DONUT_NPROC}) on GPUs ${DONUT_GPUS}"
        if [ "${DONUT_NPROC}" -gt 1 ]; then
            (
                CUDA_VISIBLE_DEVICES="${DONUT_GPUS}" \
                KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
                torchrun --standalone --nnodes=1 --nproc_per_node="${DONUT_NPROC}" \
                    main.py --stage train_donut --config "${CONFIG_FILE}"
            ) > "${LOG_DIR}/${DATASET}-donut.log" 2>&1 &
        else
            (
                CUDA_VISIBLE_DEVICES="${DONUT_GPUS}" \
                KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
                python main.py --stage train_donut --config "${CONFIG_FILE}"
            ) > "${LOG_DIR}/${DATASET}-donut.log" 2>&1 &
        fi
        DONUT_PID=$!
    else
        log "  DONUT skipped (KAGGLE2_SKIP_DONUT=1)"
        DONUT_PID=""
    fi

    # ---- TrOCR (DDP, GPUs 4-5 by default) --------------------------------
    log "  TrOCR  (DDP nproc=${TROCR_NPROC}) on GPUs ${TROCR_GPUS}"
    if [ "${TROCR_NPROC}" -gt 1 ]; then
        (
            CUDA_VISIBLE_DEVICES="${TROCR_GPUS}" \
            KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
            torchrun --standalone --nnodes=1 --nproc_per_node="${TROCR_NPROC}" \
                --master_port=29501 \
                main.py --stage train_trocr --config "${CONFIG_FILE}"
        ) > "${LOG_DIR}/${DATASET}-trocr.log" 2>&1 &
    else
        (
            CUDA_VISIBLE_DEVICES="${TROCR_GPUS}" \
            KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
            python main.py --stage train_trocr --config "${CONFIG_FILE}"
        ) > "${LOG_DIR}/${DATASET}-trocr.log" 2>&1 &
    fi
    TROCR_PID=$!

    # ---- YOLO (single GPU 6 by default) ----------------------------------
    log "  YOLO   (single) on GPU ${YOLO_GPU}"
    (
        CUDA_VISIBLE_DEVICES="${YOLO_GPU}" \
        KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
        python main.py --stage train_yolo --config "${CONFIG_FILE}"
    ) > "${LOG_DIR}/${DATASET}-yolo.log" 2>&1 &
    YOLO_PID=$!

    # ---- Wait for all three components -----------------------------------
    log "  waiting for components: DONUT=${DONUT_PID:-skipped} TrOCR=${TROCR_PID} YOLO=${YOLO_PID}"
    local FAILED=0
    if [ -n "${DONUT_PID}" ]; then
        if ! wait "${DONUT_PID}"; then
            log "  ERROR: DONUT failed"
            dump_log_on_fail "${DATASET}-donut" "${LOG_DIR}/${DATASET}-donut.log"
            FAILED=1
        fi
    fi
    if ! wait "${TROCR_PID}"; then
        log "  ERROR: TrOCR failed"
        dump_log_on_fail "${DATASET}-trocr" "${LOG_DIR}/${DATASET}-trocr.log"
        FAILED=1
    fi
    if ! wait "${YOLO_PID}"; then
        log "  ERROR: YOLO failed"
        dump_log_on_fail "${DATASET}-yolo" "${LOG_DIR}/${DATASET}-yolo.log"
        FAILED=1
    fi
    if [ "${FAILED}" -ne 0 ]; then
        log "Phase 1 (${DATASET}) FAILED; aborting sweep."
        return 1
    fi

    # ---- Manifest -------------------------------------------------------
    log "Phase 1 (${DATASET}): components converged → writing backbone manifest"
    KAGGLE2_RUN_ID="${BACKBONE_RUN_ID}" \
        python main.py --stage write_backbone_manifest --config "${CONFIG_FILE}" \
        > "${LOG_DIR}/${DATASET}-manifest.log" 2>&1
    log "Phase 1 (${DATASET}): backbone complete → runs/${BACKBONE_RUN_ID}/"
}

for DATASET in ${DATASETS}; do
    phase1_one_dataset "${DATASET}"
done

# ---------------------------------------------------------------------------
# Phase 2 — multi-seed AttentionAssigner sweep.  Each seed gets one GPU.
# Bin-pack: with 8 GPUs and 5 seeds, all run concurrently with 3 idle.
# ---------------------------------------------------------------------------

queue_assigner_jobs() {
    local DATASET="$1"
    local CONFIG_FILE
    CONFIG_FILE="$(resolve_config "${DATASET}")"
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
            wait "${PIDS[0]}" || log "WARN: PID ${PIDS[0]} returned non-zero"
            PIDS=("${PIDS[@]:1}")
        fi
    done
    log "Phase 2 (${DATASET}): waiting for trailing ${#PIDS[@]} jobs"
    for PID in "${PIDS[@]}"; do
        wait "${PID}" || log "WARN: PID ${PID} returned non-zero"
    done
    log "Phase 2 (${DATASET}): all ${#SEED_RUNS[@]} seeds complete"
    for R in "${SEED_RUNS[@]}"; do
        printf "%s\n" "${R}" >> "${LOG_DIR}/seed_runs.txt"
    done
}

for DATASET in ${DATASETS}; do
    queue_assigner_jobs "${DATASET}"
done

# ---------------------------------------------------------------------------
# Phase 4 — render the paper for each per-seed run, then aggregate.
# Paper rendering is fast (≤30 s per run) so do it sequentially to avoid
# TeX log interleaving.
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
