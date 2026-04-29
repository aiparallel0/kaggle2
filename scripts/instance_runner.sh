#!/usr/bin/env bash
# scripts/instance_runner.sh
# What each vast.ai instance runs after `vastai_bootstrap.sh`.
#
# Two modes, selected by KAGGLE2_PHASE env:
#   * KAGGLE2_PHASE=backbone — trains DONUT + YOLO + TrOCR, packs the
#     result, and uploads it to the shared bucket so the per-seed
#     workers can pull it.
#   * KAGGLE2_PHASE=assigner — pulls a pre-trained backbone, trains
#     the AttentionAssigner for one seed, runs eval + paper, packs
#     the output.
#
# Required env (set by scripts/vastai_swarm.sh on the launching host
# and forwarded to each vast.ai instance via env-string):
#   KAGGLE2_PHASE          backbone | assigner
#   KAGGLE2_RUN_ID         human-readable run id (used for upload key)
#   KAGGLE2_BUCKET_URL     rclone-compatible URL for shared storage
#                          (e.g. s3:my-bucket or gdrive:kaggle2/runs)
#   KAGGLE2_BACKBONE_KEY   bucket key for the backbone (tar.zst)
#                          — required in assigner phase only
#   KAGGLE2_SEED           integer seed for assigner phase
#   KAGGLE2_DATASET        sroie | cord | wildreceipt
#                          (controls which config to load)

set -euo pipefail

log() { printf "\033[1;36m[runner]\033[0m %s\n" "$*"; }
require() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "ERROR: $name is not set." >&2
        exit 2
    fi
}

require KAGGLE2_PHASE
require KAGGLE2_RUN_ID
require KAGGLE2_BUCKET_URL

CONFIG_FILE="${KAGGLE2_CONFIG:-configs/default.json}"
case "${KAGGLE2_DATASET:-sroie}" in
    sroie)        CONFIG_FILE="configs/default.json" ;;
    canonical)    CONFIG_FILE="configs/canonical_5seed.json" ;;
    *) log "Custom dataset: ${KAGGLE2_DATASET}; using ${CONFIG_FILE}" ;;
esac

log "Phase: ${KAGGLE2_PHASE}  Run: ${KAGGLE2_RUN_ID}  Config: ${CONFIG_FILE}"

# rclone is the path-of-least-resistance for "any bucket": S3, GCS,
# Backblaze, GDrive, R2, etc.  Install if missing.
if ! command -v rclone >/dev/null; then
    log "Installing rclone"
    curl -fsSL https://rclone.org/install.sh | bash
fi

case "${KAGGLE2_PHASE}" in
    backbone)
        log "Training shared backbone (DONUT + YOLO + TrOCR)"
        KAGGLE2_RUN_ID="${KAGGLE2_RUN_ID}" \
            python main.py --stage train_backbone --config "${CONFIG_FILE}"
        log "Packing backbone artefacts"
        BACKBONE_TAR="/tmp/${KAGGLE2_RUN_ID}-backbone.tar.zst"
        tar --use-compress-program=zstd -cf "${BACKBONE_TAR}" \
            -C "runs/${KAGGLE2_RUN_ID}" donut yolo trocr backbone_manifest.json
        log "Uploading to ${KAGGLE2_BUCKET_URL}/${KAGGLE2_RUN_ID}-backbone.tar.zst"
        rclone copyto "${BACKBONE_TAR}" \
            "${KAGGLE2_BUCKET_URL}/${KAGGLE2_RUN_ID}-backbone.tar.zst" \
            --progress
        log "Backbone phase complete"
        ;;
    assigner)
        require KAGGLE2_BACKBONE_KEY
        require KAGGLE2_SEED
        log "Pulling pre-trained backbone: ${KAGGLE2_BACKBONE_KEY}"
        BACKBONE_TAR="/tmp/${KAGGLE2_BACKBONE_KEY}"
        rclone copyto \
            "${KAGGLE2_BUCKET_URL}/${KAGGLE2_BACKBONE_KEY}" \
            "${BACKBONE_TAR}" --progress
        BACKBONE_DIR="/tmp/backbone-${KAGGLE2_SEED}"
        mkdir -p "${BACKBONE_DIR}"
        tar --use-compress-program=zstd -xf "${BACKBONE_TAR}" -C "${BACKBONE_DIR}"
        log "Training assigner (seed=${KAGGLE2_SEED})"
        KAGGLE2_RUN_ID="${KAGGLE2_RUN_ID}" \
        KAGGLE2_BACKBONE_FROM="${BACKBONE_DIR}" \
            python main.py --stage train_assigner --config "${CONFIG_FILE}" \
            --seeds "${KAGGLE2_SEED}"
        log "Running eval"
        KAGGLE2_RUN_ID="${KAGGLE2_RUN_ID}" \
            python main.py --stage eval --config "${CONFIG_FILE}" \
            --seeds "${KAGGLE2_SEED}"
        log "Rendering paper"
        KAGGLE2_RUN_ID="${KAGGLE2_RUN_ID}" \
            python main.py --stage paper --config "${CONFIG_FILE}" || true
        log "Packing per-seed artefacts"
        SEED_TAR="/tmp/${KAGGLE2_RUN_ID}.tar.zst"
        tar --use-compress-program=zstd -cf "${SEED_TAR}" -C runs "${KAGGLE2_RUN_ID}"
        log "Uploading to ${KAGGLE2_BUCKET_URL}/seeds/${KAGGLE2_RUN_ID}.tar.zst"
        rclone copyto "${SEED_TAR}" \
            "${KAGGLE2_BUCKET_URL}/seeds/${KAGGLE2_RUN_ID}.tar.zst" \
            --progress
        log "Assigner phase complete"
        ;;
    *)
        echo "ERROR: KAGGLE2_PHASE must be 'backbone' or 'assigner', got '${KAGGLE2_PHASE}'" >&2
        exit 2
        ;;
esac
