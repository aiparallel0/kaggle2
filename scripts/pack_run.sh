#!/usr/bin/env bash
# scripts/pack_run.sh — archive one runs/<run_id>/ for vast.ai → Copilot.
#
# Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
# Role: turn a complete run directory into a single self-contained
#   .tar.zst archive + sha256 sidecar, ready to scp off the vast.ai
#   box and attach to a Copilot PR review.  Uses zstd for fast, high-
#   ratio compression (tar.gz fallback if zstd is missing).  Named
#   after the run_id so a directory of archives stays sorted by run.
#
# Usage (from repo root):
#   bash scripts/pack_run.sh                 # pack the latest run
#   bash scripts/pack_run.sh 20260424T103055Z-a1b2c3d
#   bash scripts/pack_run.sh <run_id> /tmp   # custom output dir
#
# Exit codes:
#   0  archive + sha256 produced; one-line upload instruction on stdout
#   1  run directory missing or not under runs/
#   2  tar / zstd / sha256sum failed
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="$REPO_ROOT/runs"
RUN_ARG="${1:-}"
OUT_DIR="${2:-$REPO_ROOT}"

log() { printf "\033[1;36m[pack]\033[0m %s\n" "$*" >&2; }

if [ ! -d "$RUNS_ROOT" ]; then
    echo "ERROR: $RUNS_ROOT does not exist — run 'make train' first." >&2
    exit 1
fi

if [ -z "$RUN_ARG" ]; then
    RUN_ID="$(python - <<'PY'
from pathlib import Path
from core.runlayout import latest_run
p = latest_run("runs")
print(p.name if p else "")
PY
)"
    if [ -z "$RUN_ID" ]; then
        echo "ERROR: no run directories under $RUNS_ROOT" >&2
        exit 1
    fi
else
    RUN_ID="$RUN_ARG"
fi

RUN_DIR="$RUNS_ROOT/$RUN_ID"
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: $RUN_DIR does not exist" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

if command -v zstd >/dev/null 2>&1; then
    ARCHIVE="$OUT_DIR/${RUN_ID}.tar.zst"
    log "packing $RUN_DIR → $ARCHIVE (zstd)"
    if ! tar --owner=0 --group=0 -cf - -C "$RUNS_ROOT" "$RUN_ID" \
         | zstd -T0 -19 -q -o "$ARCHIVE"; then
        echo "ERROR: tar|zstd failed" >&2
        exit 2
    fi
else
    ARCHIVE="$OUT_DIR/${RUN_ID}.tar.gz"
    log "packing $RUN_DIR → $ARCHIVE (gzip; zstd not found)"
    if ! tar --owner=0 --group=0 -czf "$ARCHIVE" -C "$RUNS_ROOT" "$RUN_ID"; then
        echo "ERROR: tar failed" >&2
        exit 2
    fi
fi

CHECKSUM="${ARCHIVE}.sha256"
if command -v sha256sum >/dev/null 2>&1; then
    (cd "$OUT_DIR" && sha256sum "$(basename "$ARCHIVE")" > "$CHECKSUM")
elif command -v shasum >/dev/null 2>&1; then
    (cd "$OUT_DIR" && shasum -a 256 "$(basename "$ARCHIVE")" > "$CHECKSUM")
else
    echo "ERROR: neither sha256sum nor shasum available" >&2
    exit 2
fi

SIZE="$(du -h "$ARCHIVE" | awk '{print $1}')"
log "archive: $ARCHIVE  ($SIZE)"
log "sha256:  $CHECKSUM"

# One-line upload instruction — kept flat so operators can copy-paste
# without the terminal reflowing it into multiple lines.
echo "UPLOAD: attach $ARCHIVE (sha256: $(awk '{print $1}' "$CHECKSUM")) to the PR review thread."
