#!/usr/bin/env bash
# kaggle2/scripts/resume_after_trocr.sh
#
# Disk-full recovery driver for a run that crashed at
#   File "/workspace/kaggle2/models/trocr_train.py", line 158, in train_trocr
#     trainer.save_model(out_dir)
# ...because the container disk filled up while transformers tried to
# serialise the best TrOCR checkpoint on top of per-epoch saves.
#
# What this script does — no re-training of DONUT / YOLO / TrOCR required:
#   1. Aggressively frees disk on the vast.ai container (apt + pip caches,
#      /tmp, __pycache__, ~/.cache/huggingface/hub blob de-dup is left
#      alone so processor reload does not re-download).
#   2. Invokes scripts/resume_after_trocr.py to:
#        - promote the best per-epoch TrOCR checkpoint into results/trocr/
#        - resave the TrOCRProcessor that never got written
#        - run the train_assigner step that main.py never reached
#        - write results/pipeline_meta.json
#   3. Runs `python main.py --stage eval`  (uses all four now-present models)
#   4. Runs `python main.py --stage paper` (compiles report/paper_filled.pdf)
#
# Idempotent — safe to rerun if any later step fails.
#
# Usage on the same vast.ai instance that crashed:
#   cd /workspace/kaggle2
#   bash scripts/resume_after_trocr.sh
#
# Environment knobs:
#   PYTHON         python interpreter (default: python)
#   CONFIG         config.json path (default: config.json)
#   SKIP_CLEANUP=1 keep the DONUT checkpoint-* dirs and yolo_data/ (fast
#                  re-run after the first cleanup already freed disk).

set -euo pipefail
if [ "${VERBOSE:-0}" = "1" ]; then set -x; fi

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-config.json}"
SKIP_CLEANUP="${SKIP_CLEANUP:-0}"

log() { printf '\033[1;36m[resume]\033[0m %s\n' "$*"; }

if [ ! -f "$CONFIG" ]; then
    echo "ERROR: CONFIG=$CONFIG not found. Run this script from the repo root." >&2
    exit 1
fi

log "Container disk before cleanup:"
df -h / 2>/dev/null | tail -n +1 || true
df -h . 2>/dev/null | tail -n +1 || true

if [ "$SKIP_CLEANUP" != "1" ]; then
    log "Purging apt and pip caches"
    apt-get clean >/dev/null 2>&1 || true
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true
    "$PYTHON" -m pip cache purge >/dev/null 2>&1 || true
    rm -rf /root/.cache/pip /home/*/.cache/pip 2>/dev/null || true

    log "Clearing /tmp build/download residue"
    # Be conservative: only remove obvious transient files, never /tmp wholesale
    # (vast.ai sometimes uses /tmp for jupyter sockets the user is relying on).
    find /tmp -maxdepth 2 \( \
        -name 'pip-*' -o \
        -name 'tmp*.whl' -o \
        -name '*.tar.gz' -o \
        -name 'tectonic-*' -o \
        -name 'torch-*.bin' \
    \) -exec rm -rf {} + 2>/dev/null || true

    log "Clearing __pycache__ trees in the repo"
    find . -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
fi

log "Promoting best TrOCR checkpoint + running remaining train steps"
if [ "$SKIP_CLEANUP" = "1" ]; then
    "$PYTHON" scripts/resume_after_trocr.py --config "$CONFIG" --skip-cleanup
else
    "$PYTHON" scripts/resume_after_trocr.py --config "$CONFIG"
fi

log "Container disk after resume-train:"
df -h . 2>/dev/null | tail -n +1 || true

log "Running eval stage"
"$PYTHON" main.py --stage eval --config "$CONFIG"

log "Running paper stage (tectonic/pdflatex; .tex emitted regardless)"
if ! "$PYTHON" main.py --stage paper --config "$CONFIG"; then
    log "paper stage exited non-zero — report/paper_filled.tex should still be present."
fi

log "Done.  Artifacts:"
log "  results/donut/"
log "  results/yolo/run/weights/best.pt"
log "  results/trocr/"
log "  results/assigner.pt"
log "  results/combined_metrics.json"
log "  report/paper_filled.{tex,pdf}"
