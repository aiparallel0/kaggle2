#!/usr/bin/env bash
# scripts/vastai_swarm.sh
# Launches a parallel multi-seed × multi-dataset sweep on vast.ai.
#
# Layout (one big H100 instance + N small instances, all parallel):
#
#   1. ONE H100/H200 instance trains the shared backbone
#      (DONUT + YOLO + TrOCR) and uploads it to a shared bucket.
#      Wall clock on 1× H100 SXM5: ~6–8 min with batch-size scaling.
#
#   2. N small instances (one per seed × dataset) pull the backbone,
#      train ONLY the AttentionAssigner (~2 min on H100 / ~30 s on
#      H200), run eval, render paper, and upload per-seed results.
#
# Total wall clock on 8× H100 SXM5 backbone + 10× H100 PCIe per-seed:
#   max(backbone, per-seed) = max(7 min, 4 min) = ~7 min wall-clock
#   per (seed, dataset).  Across 10 (seed, dataset) tuples in
#   parallel: still ~7 min wall-clock, since they run concurrently.
#
# Money: ~$2/hr × 8-GPU H100 instance × 10 min = $0.30 backbone;
#   ~$0.50/hr × H100 PCIe × 4 min × 10 instances = $0.35.
#   Total ~$0.65/sweep at vast.ai mid-2026 spot rates.  Adjust the
#   GPU class env vars below for your wallet.
#
# Usage (from a laptop with `vastai` CLI installed and configured):
#
#   export KAGGLE2_BUCKET_URL="s3:my-kaggle2-bucket"
#   export KAGGLE2_RCLONE_CONF="$HOME/.config/rclone/rclone.conf"
#   bash scripts/vastai_swarm.sh "42 1 2 3 5" "sroie canonical"
#
# The first arg is the seed list, the second is the dataset list.
# Both default to a 5-seed × 1-dataset sweep.
#
# IMPORTANT: this script does not mutate or break the local
# RTX-4090 path.  ``make all`` continues to work as before.

set -euo pipefail

SEEDS="${1:-42 1 2 3 5}"
DATASETS="${2:-canonical}"
SWEEP_ID="${KAGGLE2_SWEEP_ID:-sweep-$(date -u +%Y%m%dT%H%M%SZ)}"

log() { printf "\033[1;36m[swarm]\033[0m %s\n" "$*"; }
require() {
    local name="$1"
    if [ -z "${!name:-}" ]; then
        echo "ERROR: $name must be set in your environment." >&2
        echo "       Add it to your shell rc or pass it inline." >&2
        exit 2
    fi
}

require KAGGLE2_BUCKET_URL
require KAGGLE2_RCLONE_CONF
if ! command -v vastai >/dev/null; then
    echo "ERROR: vastai CLI not found. pip install vastai" >&2
    exit 2
fi

# Default GPU classes — tune to your budget.
BACKBONE_GPU="${KAGGLE2_BACKBONE_GPU:-RTX_4090}"   # any 24 GB+ GPU
ASSIGNER_GPU="${KAGGLE2_ASSIGNER_GPU:-RTX_4090}"   # any 16 GB+ GPU
IMAGE="${KAGGLE2_DOCKER_IMAGE:-pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime}"
DISK_GB="${KAGGLE2_DISK_GB:-60}"

# Repo source: the swarm clones from GitHub on each instance, so
# the per-instance bootstrap runs exactly the code that's pushed.
GIT_URL="${KAGGLE2_GIT_URL:-https://github.com/aiparallel0/kaggle2.git}"
GIT_BRANCH="${KAGGLE2_GIT_BRANCH:-main}"

# Onstart script common to both phases.  Streams from $1 (rclone
# config) into the instance, clones the repo, runs the bootstrap.
mk_onstart() {
    local phase="$1" extra_env="$2"
    cat <<ONSTART
#!/usr/bin/env bash
set -euo pipefail
mkdir -p /root/.config/rclone
cat > /root/.config/rclone/rclone.conf <<RCLONE
$(cat "${KAGGLE2_RCLONE_CONF}")
RCLONE
cd /workspace
if [ ! -d kaggle2 ]; then
    git clone --depth=1 -b "${GIT_BRANCH}" "${GIT_URL}"
fi
cd kaggle2
bash scripts/vastai_bootstrap.sh
export KAGGLE2_PHASE="${phase}"
${extra_env}
exec bash scripts/instance_runner.sh
ONSTART
}

# Phase 1: launch the backbone instance and BLOCK until its key
# appears in the bucket.  Subsequent phase 2 instances pull this key.
BACKBONE_KEY="${SWEEP_ID}-backbone.tar.zst"
log "Phase 1: backbone instance (gpu=${BACKBONE_GPU}, key=${BACKBONE_KEY})"
BACKBONE_ENV=$(cat <<ENV
export KAGGLE2_RUN_ID="${SWEEP_ID}-backbone"
export KAGGLE2_BUCKET_URL="${KAGGLE2_BUCKET_URL}"
export KAGGLE2_DATASET="$(echo $DATASETS | awk '{print $1}')"
ENV
)
ONSTART_BACKBONE="$(mk_onstart backbone "${BACKBONE_ENV}")"
BACKBONE_OFFER=$(vastai search offers \
    "gpu_name=${BACKBONE_GPU} num_gpus>=1 reliability>0.95" \
    --raw 2>/dev/null | python -c "import sys, json; offers=json.load(sys.stdin); print(min(offers, key=lambda x: x['dph_total'])['id'])")
log "Backbone offer: ${BACKBONE_OFFER}"
BACKBONE_INSTANCE=$(vastai create instance "${BACKBONE_OFFER}" \
    --image "${IMAGE}" \
    --disk "${DISK_GB}" \
    --onstart-cmd "${ONSTART_BACKBONE}" \
    --raw | python -c "import sys, json; print(json.load(sys.stdin)['new_contract'])")
log "Backbone instance ID: ${BACKBONE_INSTANCE}"

# Poll the bucket for the backbone tar.
log "Waiting for backbone upload (typical: 6-10 min on 8x H100)..."
while ! rclone --config "${KAGGLE2_RCLONE_CONF}" \
    lsf "${KAGGLE2_BUCKET_URL}" 2>/dev/null \
    | grep -q "^${BACKBONE_KEY}$"; do
    sleep 30
    log "  still waiting…"
done
log "Backbone available at ${KAGGLE2_BUCKET_URL}/${BACKBONE_KEY}"
log "Destroying backbone instance ${BACKBONE_INSTANCE}"
vastai destroy instance "${BACKBONE_INSTANCE}" || true

# Phase 2: fan out one instance per (seed, dataset) tuple.
INSTANCE_IDS=()
for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        SEED_RUN_ID="${SWEEP_ID}-${DATASET}-seed${SEED}"
        log "Phase 2: launching ${SEED_RUN_ID} (gpu=${ASSIGNER_GPU})"
        SEED_ENV=$(cat <<ENV
export KAGGLE2_RUN_ID="${SEED_RUN_ID}"
export KAGGLE2_BUCKET_URL="${KAGGLE2_BUCKET_URL}"
export KAGGLE2_BACKBONE_KEY="${BACKBONE_KEY}"
export KAGGLE2_SEED="${SEED}"
export KAGGLE2_DATASET="${DATASET}"
ENV
)
        ONSTART_SEED="$(mk_onstart assigner "${SEED_ENV}")"
        OFFER_ID=$(vastai search offers \
            "gpu_name=${ASSIGNER_GPU} num_gpus>=1 reliability>0.95" \
            --raw 2>/dev/null | python -c "import sys, json; offers=json.load(sys.stdin); print(min(offers, key=lambda x: x['dph_total'])['id'])")
        INST=$(vastai create instance "${OFFER_ID}" \
            --image "${IMAGE}" \
            --disk "${DISK_GB}" \
            --onstart-cmd "${ONSTART_SEED}" \
            --raw | python -c "import sys, json; print(json.load(sys.stdin)['new_contract'])")
        log "  instance ${INST}"
        INSTANCE_IDS+=("${INST}")
    done
done

# Wait for every per-seed run to upload its tarball.
log "Waiting for ${#INSTANCE_IDS[@]} per-seed uploads (typical: 3-5 min)..."
EXPECTED_SEEDS=()
for DATASET in $DATASETS; do
    for SEED in $SEEDS; do
        EXPECTED_SEEDS+=("${SWEEP_ID}-${DATASET}-seed${SEED}.tar.zst")
    done
done
while true; do
    UPLOADED=$(rclone --config "${KAGGLE2_RCLONE_CONF}" \
        lsf "${KAGGLE2_BUCKET_URL}/seeds/" 2>/dev/null | sort -u || true)
    MISSING=0
    for KEY in "${EXPECTED_SEEDS[@]}"; do
        if ! echo "${UPLOADED}" | grep -q "^${KEY}$"; then
            MISSING=$((MISSING + 1))
        fi
    done
    if [ "${MISSING}" -eq 0 ]; then
        break
    fi
    log "  ${MISSING}/${#EXPECTED_SEEDS[@]} still pending…"
    sleep 30
done

log "All seeds uploaded.  Tearing down ${#INSTANCE_IDS[@]} instances."
for INST in "${INSTANCE_IDS[@]}"; do
    vastai destroy instance "${INST}" || true
done

log "Sweep ${SWEEP_ID} complete."
log "Pull all results locally:"
log "  rclone --config ${KAGGLE2_RCLONE_CONF} copy ${KAGGLE2_BUCKET_URL}/seeds ./runs/${SWEEP_ID}"
log "Aggregate:"
log "  python scripts/aggregate_seeds.py ./runs/${SWEEP_ID}/*"
