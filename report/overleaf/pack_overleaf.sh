#!/usr/bin/env bash
# kaggle2/report/overleaf/pack_overleaf.sh
#
# Bundle a completed run into a single Overleaf-ready .zip.
#
# Usage:
#   bash report/overleaf/pack_overleaf.sh runs/<run_id>
#
# Produces:
#   runs/<run_id>/overleaf_<variant>.zip
#
# The .zip contains paper_filled.tex, references.bib, every figure
# under figures/ and the run root, the entire report/sections/
# directory (so individual section edits work in Overleaf), and the
# MANIFEST.json so reviewers can verify provenance.
set -euo pipefail

if [ "${1:-}" = "" ]; then
    echo "usage: $0 runs/<run_id>" >&2
    exit 2
fi
RUN_DIR="$1"
if [ ! -d "$RUN_DIR" ]; then
    echo "error: $RUN_DIR is not a directory" >&2
    exit 2
fi
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PAPER_TEX="$RUN_DIR/paper/paper_filled.tex"
if [ ! -f "$PAPER_TEX" ]; then
    echo "error: $PAPER_TEX not found — run 'make paper' first." >&2
    exit 2
fi

# Detect variant from combined_metrics.json so the zip name matches.
VARIANT="advanced"
if [ -f "$RUN_DIR/combined_metrics.json" ]; then
    if grep -q '"test_set_kind": *"internal_63"' "$RUN_DIR/combined_metrics.json"; then
        VARIANT="basic"
    fi
fi

STAGE="$RUN_DIR/_overleaf_stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/figures" "$STAGE/sections"

cp "$PAPER_TEX" "$STAGE/paper_filled.tex"
cp "$REPO_ROOT/report/references.bib" "$STAGE/references.bib"
cp -r "$REPO_ROOT/report/sections/." "$STAGE/sections/"
# Figures: both runs/<id>/figures/ and any flat runs/<id>/*.pdf.
if [ -d "$RUN_DIR/figures" ]; then
    cp "$RUN_DIR/figures/"*.pdf "$STAGE/figures/" 2>/dev/null || true
fi
cp "$RUN_DIR/"*.pdf "$STAGE/figures/" 2>/dev/null || true
# Provenance.
[ -f "$RUN_DIR/MANIFEST.json" ] && cp "$RUN_DIR/MANIFEST.json" "$STAGE/MANIFEST.json"

ZIP_OUT="$RUN_DIR/overleaf_${VARIANT}.zip"
rm -f "$ZIP_OUT"
( cd "$STAGE" && zip -qr "$ZIP_OUT" . )
rm -rf "$STAGE"
echo "wrote $ZIP_OUT (variant=$VARIANT)"
