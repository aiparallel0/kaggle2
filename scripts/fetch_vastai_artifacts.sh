#!/usr/bin/env bash
# kaggle2/scripts/fetch_vastai_artifacts.sh
#
# Pull the artefacts of a finished (or crashed) kaggle2 run off a vast.ai
# instance so we can post-mortem the F1 locally without keeping the GPU
# machine around.
#
# What this script downloads, in order of priority:
#   1. results/combined_metrics.json      -- headline numbers
#   2. results/donut_metrics.json         -- DONUT per-field F1
#   3. results/pipeline_metrics.json      -- pipeline per-field F1
#   4. results/assigner_metrics.json      -- assigner best val loss
#   5. results/split.json                 -- the persisted 500/63/63 split
#   6. results/pipeline_meta.json         -- yolo_img_size used at eval
#   7. results/yolo/run/**                -- YOLO train logs, best.pt, results.png
#   8. results/donut/*.json + donut_path  -- DONUT metadata (no checkpoint blobs)
#   9. results/trocr/*.json               -- TrOCR processor + trainer_state
#  10. results/assigner.pt                -- 50-KB assigner checkpoint
#  11. report/paper_filled.{tex,pdf}      -- final paper
#  12. recent nohup/stdout logs           -- root cause of any crash
#
# Heavy model weights (donut/model.safetensors, trocr/model.safetensors) are
# OPTIONAL and off by default (they're 500 MB and 1.4 GB respectively).
# Pass --with-weights to include them.
#
# Connection:
#   Pass the SSH target the same way vast.ai gives it to you.
#   Examples:
#     scripts/fetch_vastai_artifacts.sh -p 12345 root@ssh4.vast.ai
#     scripts/fetch_vastai_artifacts.sh root@12.34.56.78
#     scripts/fetch_vastai_artifacts.sh --ssh-cmd "ssh -p 12345 root@ssh4.vast.ai"
#
# The remote repo path defaults to /workspace/kaggle2 (the path baked into
# scripts/vastai_bootstrap.sh). Override with --remote-root.
#
# Local destination defaults to ./vastai_dump/<ISO timestamp>/ so repeated
# fetches don't overwrite each other.
set -euo pipefail

REMOTE_ROOT="/workspace/kaggle2"
LOCAL_ROOT="./vastai_dump/$(date -u +%Y%m%dT%H%M%SZ)"
SSH_PORT=""
SSH_TARGET=""
SSH_CMD=""
WITH_WEIGHTS=0
DRY_RUN=0

usage() {
    sed -n '2,35p' "$0"
    exit "${1:-0}"
}

while [ $# -gt 0 ]; do
    case "$1" in
        -p|--port)         SSH_PORT="$2"; shift 2 ;;
        --ssh-cmd)         SSH_CMD="$2"; shift 2 ;;
        --remote-root)     REMOTE_ROOT="$2"; shift 2 ;;
        --local-root)      LOCAL_ROOT="$2"; shift 2 ;;
        --with-weights)    WITH_WEIGHTS=1; shift ;;
        --dry-run)         DRY_RUN=1; shift ;;
        -h|--help)         usage 0 ;;
        -*)                echo "Unknown flag: $1" >&2; usage 1 ;;
        *)                 SSH_TARGET="$1"; shift ;;
    esac
done

if [ -z "$SSH_CMD" ]; then
    if [ -z "$SSH_TARGET" ]; then
        echo "ERROR: provide user@host (and optionally -p PORT) or --ssh-cmd '...'." >&2
        usage 1
    fi
    SSH_CMD="ssh -o StrictHostKeyChecking=accept-new"
    [ -n "$SSH_PORT" ] && SSH_CMD="$SSH_CMD -p $SSH_PORT"
    SSH_CMD="$SSH_CMD $SSH_TARGET"
fi

# rsync uses the same connection args as ssh; strip the trailing target so it
# becomes a pure "-e ssh ..." string.
SSH_BIN="${SSH_CMD% *}"
SSH_HOST="${SSH_CMD##* }"

say() { printf '\033[1;36m[fetch]\033[0m %s\n' "$*"; }
run() { if [ "$DRY_RUN" = "1" ]; then echo "+ $*"; else eval "$@"; fi; }

mkdir -p "$LOCAL_ROOT"
say "Remote: $SSH_HOST"
say "Remote root: $REMOTE_ROOT"
say "Local dest: $LOCAL_ROOT"
say "With model weights: $([ "$WITH_WEIGHTS" = "1" ] && echo yes || echo no)"

# Rsync include/exclude pattern list. Order matters: include dirs before the
# files inside them, deny-list weights unless --with-weights.
INCLUDES=(
    "results/"
    "results/*.json"
    "results/yolo/"
    "results/yolo/run/"
    "results/yolo/run/**"
    "results/donut/"
    "results/donut/*.json"
    "results/donut/generation_config.json"
    "results/donut/config.json"
    "results/donut/preprocessor_config.json"
    "results/donut/tokenizer*"
    "results/donut/special_tokens_map.json"
    "results/donut/added_tokens.json"
    "results/trocr/"
    "results/trocr/*.json"
    "results/trocr/preprocessor_config.json"
    "results/trocr/tokenizer*"
    "results/trocr/vocab.json"
    "results/trocr/merges.txt"
    "results/trocr/special_tokens_map.json"
    "results/assigner.pt"
    "report/"
    "report/paper_filled.tex"
    "report/paper_filled.pdf"
)
EXCLUDES=(
    "results/yolo_data/"        # staging mirror of SROIE images, huge, regen
    "results/yolo/run/weights/last.pt"
    "results/**/checkpoint-*"   # per-epoch checkpoints — training artefacts only
)
if [ "$WITH_WEIGHTS" = "1" ]; then
    INCLUDES+=(
        "results/donut/model.safetensors"
        "results/donut/pytorch_model.bin"
        "results/trocr/model.safetensors"
        "results/trocr/pytorch_model.bin"
        "results/yolo/run/weights/best.pt"
    )
else
    EXCLUDES+=(
        "results/donut/model.safetensors"
        "results/donut/pytorch_model.bin"
        "results/trocr/model.safetensors"
        "results/trocr/pytorch_model.bin"
        "results/yolo/run/weights/*.pt"
    )
fi
# Deny everything not explicitly matched.
EXCLUDES+=("*")

FILTER_ARGS=()
for p in "${INCLUDES[@]}"; do FILTER_ARGS+=(--include="$p"); done
for p in "${EXCLUDES[@]}"; do FILTER_ARGS+=(--exclude="$p"); done

say "rsync --dry-run preview:"
run rsync -avh --dry-run -e "'$SSH_BIN'" \
    "${FILTER_ARGS[@]}" \
    "$SSH_HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/" | head -40 || true

if [ "$DRY_RUN" = "1" ]; then
    say "Dry run only — exiting before real rsync."
    exit 0
fi

say "Fetching artefacts..."
rsync -avh --partial --progress -e "$SSH_BIN" \
    "${FILTER_ARGS[@]}" \
    "$SSH_HOST:$REMOTE_ROOT/" "$LOCAL_ROOT/"

# Snapshot nohup-style stdout/stderr logs and the resume script's breadcrumbs.
# None of these paths are required (they may not exist) — tolerate misses.
say "Fetching loose logs (best-effort, errors ignored)..."
for pattern in \
    "$REMOTE_ROOT/nohup.out" \
    "$REMOTE_ROOT/*.log" \
    "$REMOTE_ROOT/results/*.log" \
    "$REMOTE_ROOT/results/**/events.out.tfevents.*"; do
    rsync -avh --ignore-missing-args -e "$SSH_BIN" \
        "$SSH_HOST:$pattern" "$LOCAL_ROOT/logs/" 2>/dev/null || true
done

# Also capture a small remote "state" snapshot — disk usage, Python/torch
# versions, GPU, and the directory listing — so we can tell at a glance
# what machine produced these numbers.
say "Capturing remote environment snapshot..."
$SSH_BIN "$SSH_HOST" "bash -lc '
    echo == date ==;        date -u;
    echo == uname ==;       uname -a;
    echo == df ==;          df -h $REMOTE_ROOT || true;
    echo == nvidia-smi ==;  nvidia-smi || true;
    echo == python ==;      python --version 2>&1;
    echo == torch ==;       python -c \"import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())\" 2>&1 || true;
    echo == git HEAD ==;    (cd $REMOTE_ROOT && git rev-parse HEAD && git status --porcelain);
    echo == ls results ==;  ls -la $REMOTE_ROOT/results 2>&1 | head -200;
'" > "$LOCAL_ROOT/remote_env.txt" 2>&1 || true

say "Done."
say "Local dump:         $LOCAL_ROOT"
say "Headline metrics:   $LOCAL_ROOT/results/combined_metrics.json"
say "Env snapshot:       $LOCAL_ROOT/remote_env.txt"
say "Paper:              $LOCAL_ROOT/report/paper_filled.pdf"
