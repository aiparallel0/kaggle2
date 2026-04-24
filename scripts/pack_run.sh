#!/usr/bin/env bash
# scripts/pack_run.sh — archive one runs/<run_id>/ for vast.ai → Copilot.
#
# Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
# Role: turn a complete run directory into a single .tar.zst archive
#   + sha256 sidecar (tar.gz fallback if zstd is missing).  Named
#   after the run_id so a directory of archives stays sorted by run.
#
# Modes:
#   --light (default)  Exclude any file > LIGHT_MAX_BYTES (1 MiB) and
#                      heavy checkpoint subdirs (donut/, trocr/,
#                      yolo/run/, yolo_data/).  Kept: JSON sidecars,
#                      figure PDFs, paper LaTeX/PDF, logs, manifest —
#                      the reviewer-useful outputs.  Typical: a few MiB.
#   --full             Include everything (model weights: DONUT ~770,
#                      TrOCR ~300, YOLO ~6 MiB).
#
# Usage (from repo root):
#   bash scripts/pack_run.sh                           # latest, light
#   bash scripts/pack_run.sh --full                    # latest, full
#   bash scripts/pack_run.sh 20260424T103055Z-a1b2c3d  # specific, light
#   bash scripts/pack_run.sh --full <run_id> /tmp      # full, custom outdir
#
# Exit codes: 0 ok; 1 run dir missing; 2 tar/zstd/sha256sum failed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="$REPO_ROOT/runs"

MODE="light"
POSITIONAL=()
for arg in "$@"; do
    case "$arg" in
        --light) MODE="light" ;;
        --full)  MODE="full" ;;
        --help|-h) sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
        *) POSITIONAL+=("$arg") ;;
    esac
done
set -- "${POSITIONAL[@]+"${POSITIONAL[@]}"}"
RUN_ARG="${1:-}"
OUT_DIR="${2:-$REPO_ROOT}"

# Files larger than this (bytes) are excluded in --light mode.  1 MiB
# lets every JSON/figure/log through; only model checkpoints are dropped.
LIGHT_MAX_BYTES="${LIGHT_MAX_BYTES:-1048576}"
# Whole-directory excludes in --light mode (HF safetensors, YOLO weights
# + dataset mirror).  Kept grouped so tar's --exclude-from stays readable.
HEAVY_DIRS=("donut" "trocr" "yolo/run" "yolo_data")

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

# --- Build the exclusion list ---------------------------------------------
# In --light mode: emit (a) ``EXCLUDED.txt`` inside the run_dir (gets
# archived so reviewers see what was stripped) and (b) a tar
# --exclude-from file.  In --full mode a stub EXCLUDED.txt keeps the
# schema uniform.
EXCLUDED_TXT="$RUN_DIR/EXCLUDED.txt"
EXCLUDE_FROM="$(mktemp)"
trap 'rm -f "$EXCLUDE_FROM"' EXIT

if [ "$MODE" = "light" ]; then
    log "mode: light (exclude files > $((LIGHT_MAX_BYTES / 1024 / 1024)) MiB and heavy dirs)"
    {
        echo "# EXCLUDED.txt — files/dirs dropped from this archive by --light mode."
        echo "# Regenerate with 'bash scripts/pack_run.sh --full <run_id>'."
        echo "# run_id: $RUN_ID   threshold: $LIGHT_MAX_BYTES bytes"
        echo
        echo "## heavy directories (recursive)"
        for d in "${HEAVY_DIRS[@]}"; do
            if [ -d "$RUN_DIR/$d" ]; then
                printf "  %-40s  %s\n" "$d/" \
                    "$(du -sh "$RUN_DIR/$d" 2>/dev/null | awk '{print $1}')"
                printf "%s/%s/*\n" "$RUN_ID" "$d" >> "$EXCLUDE_FROM"
            fi
        done
        echo
        echo "## individual files > threshold (outside heavy dirs)"
        PRUNE_ARGS=()
        for d in "${HEAVY_DIRS[@]}"; do
            PRUNE_ARGS+=(-path "$RUN_DIR/$d" -prune -o)
        done
        while IFS= read -r -d '' f; do
            rel="${f#"$RUN_DIR/"}"
            size="$(stat -c %s "$f" 2>/dev/null || stat -f %z "$f")"
            printf "  %-40s  %10d bytes\n" "$rel" "$size"
            printf "%s/%s\n" "$RUN_ID" "$rel" >> "$EXCLUDE_FROM"
        done < <(find "$RUN_DIR" "${PRUNE_ARGS[@]}" \
                      -type f -size "+${LIGHT_MAX_BYTES}c" -print0 | sort -z)
    } > "$EXCLUDED_TXT"
    EXCLUDED_COUNT="$(grep -c '^  ' "$EXCLUDED_TXT" || true)"
    log "excluded $EXCLUDED_COUNT path(s); listing at $EXCLUDED_TXT"
else
    log "mode: full (include every file — archive may be very large)"
    printf '# EXCLUDED.txt — full-mode archive; no files dropped.\n# run_id: %s\n' \
        "$RUN_ID" > "$EXCLUDED_TXT"
fi

TAR_EXCLUDE=(--exclude-from="$EXCLUDE_FROM")

if command -v zstd >/dev/null 2>&1; then
    ARCHIVE="$OUT_DIR/${RUN_ID}.tar.zst"
    log "packing $RUN_DIR → $ARCHIVE (zstd, $MODE)"
    if ! tar --owner=0 --group=0 "${TAR_EXCLUDE[@]}" \
         -cf - -C "$RUNS_ROOT" "$RUN_ID" \
         | zstd -T0 -19 -q -f -o "$ARCHIVE"; then
        echo "ERROR: tar|zstd failed" >&2; exit 2
    fi
else
    ARCHIVE="$OUT_DIR/${RUN_ID}.tar.gz"
    log "packing $RUN_DIR → $ARCHIVE (gzip; zstd not found; $MODE)"
    if ! tar --owner=0 --group=0 "${TAR_EXCLUDE[@]}" \
         -czf "$ARCHIVE" -C "$RUNS_ROOT" "$RUN_ID"; then
        echo "ERROR: tar failed" >&2; exit 2
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
