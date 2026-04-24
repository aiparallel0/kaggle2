#!/usr/bin/env bash
# scripts/unpack_run.sh — restore a packed runs/<run_id>/ archive.
#
# Project: kaggle2 — End-to-End vs. Pipeline Receipt KIE on SROIE.
# Role: inverse of scripts/pack_run.sh.  Verifies the sha256 sidecar,
#   extracts the archive into ``runs/``, and then revalidates every
#   per-file sha256 against ``MANIFEST.json`` so reviewers detect any
#   corruption at upload time rather than during paper-building.
#
# Usage (from repo root):
#   bash scripts/unpack_run.sh <archive>         # uses sibling .sha256
#   bash scripts/unpack_run.sh <archive> --no-verify-manifest
#
# Exit codes:
#   0  archive extracted; manifest verified (unless --no-verify-manifest)
#   1  archive, sidecar, or manifest missing
#   2  sha256 mismatch (archive-level or manifest-level)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNS_ROOT="$REPO_ROOT/runs"
ARCHIVE="${1:-}"
MODE="${2:-}"

log() { printf "\033[1;36m[unpack]\033[0m %s\n" "$*" >&2; }

if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then
    echo "ERROR: archive path missing or not a file: $ARCHIVE" >&2
    exit 1
fi

CHECKSUM="${ARCHIVE}.sha256"
if [ ! -f "$CHECKSUM" ]; then
    echo "ERROR: companion sha256 sidecar not found: $CHECKSUM" >&2
    exit 1
fi

log "verifying archive sha256"
(cd "$(dirname "$ARCHIVE")" && \
    (sha256sum --check "$(basename "$CHECKSUM")" 2>/dev/null \
     || shasum -a 256 --check "$(basename "$CHECKSUM")")) \
    || { echo "ERROR: archive sha256 mismatch" >&2; exit 2; }

mkdir -p "$RUNS_ROOT"

case "$ARCHIVE" in
    *.tar.zst)
        if ! command -v zstd >/dev/null 2>&1; then
            echo "ERROR: zstd not installed (needed to unpack .tar.zst)" >&2
            exit 1
        fi
        log "extracting $ARCHIVE → $RUNS_ROOT (zstd)"
        zstd -d -c "$ARCHIVE" | tar -xf - -C "$RUNS_ROOT"
        ;;
    *.tar.gz|*.tgz)
        log "extracting $ARCHIVE → $RUNS_ROOT (gzip)"
        tar -xzf "$ARCHIVE" -C "$RUNS_ROOT"
        ;;
    *)
        echo "ERROR: unsupported archive extension: $ARCHIVE" >&2
        exit 1
        ;;
esac

# Derive the run_id from the archive filename — pack_run.sh always
# names archives ``<run_id>.tar.{zst,gz}`` so this is unambiguous.
RUN_ID="$(basename "$ARCHIVE")"
RUN_ID="${RUN_ID%.tar.zst}"
RUN_ID="${RUN_ID%.tar.gz}"
RUN_ID="${RUN_ID%.tgz}"
RUN_DIR="$RUNS_ROOT/$RUN_ID"

if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: extracted tree missing $RUN_DIR" >&2
    exit 2
fi

if [ "$MODE" = "--no-verify-manifest" ]; then
    log "skipping per-file manifest verification (--no-verify-manifest)"
    log "unpacked: $RUN_DIR"
    exit 0
fi

if [ ! -f "$RUN_DIR/MANIFEST.json" ]; then
    log "WARN: no MANIFEST.json under $RUN_DIR (older run?) — skipping per-file check"
    log "unpacked: $RUN_DIR"
    exit 0
fi

log "verifying per-file sha256 against MANIFEST.json"
python - "$RUN_DIR" <<'PY'
import hashlib, json, sys
from pathlib import Path
run_dir = Path(sys.argv[1])
manifest = json.loads((run_dir / "MANIFEST.json").read_text())
bad = 0
for entry in manifest["entries"]:
    p = run_dir / entry["relpath"]
    if not p.is_file():
        print(f"MISSING: {entry['relpath']}")
        bad += 1
        continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    if h.hexdigest() != entry["sha256"]:
        print(f"MISMATCH: {entry['relpath']}")
        bad += 1
if bad:
    print(f"FAILED: {bad} file(s) failed verification", file=sys.stderr)
    sys.exit(2)
print(f"OK: {len(manifest['entries'])} files verified")
PY

log "unpacked + verified: $RUN_DIR"
