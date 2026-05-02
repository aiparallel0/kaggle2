#!/usr/bin/env bash
# Pack the visual-first paper + presentation into a shareable zip.
# Output: paper_zip_build/focus_paper_visuals.zip
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

OUT_DIR="paper_zip_build"
PKG="$OUT_DIR/focus_paper_visuals"
ZIP="$OUT_DIR/focus_paper_visuals.zip"

rm -rf "$PKG" "$ZIP"
mkdir -p "$PKG/results/figures" \
         "$PKG/report/sections" \
         "$PKG/report" \
         "$PKG/presentation" \
         "$PKG/scripts"

# Regenerate figures so the zip is always self-consistent.
python3 scripts/build_visuals.py

# Copy paper sources.
cp paper_visual.tex                          "$PKG/"
cp report/template.tex                       "$PKG/report/"
cp report/references.bib                     "$PKG/report/" 2>/dev/null || true
cp report/sections/diagrams_focus.tex        "$PKG/report/sections/"
cp report/sections/explainer_basics.tex      "$PKG/report/sections/"

# Copy figures (PDF + PNG).
cp results/figures/*.pdf "$PKG/results/figures/"
cp results/figures/*.png "$PKG/results/figures/"

# Copy real-data fixtures so reviewers can re-run build_visuals.py.
mkdir -p "$PKG/results"
cp results/bug_timeline.json                 "$PKG/results/"
cp results/sroie_task3_competitors.json      "$PKG/results/"
cp results/foundation_baseline.json          "$PKG/results/"

# Presentation.
cp presentation/slides.tex                   "$PKG/presentation/"

# Build script (so reviewers can regenerate).
cp scripts/build_visuals.py                  "$PKG/scripts/"

# README for the zip.
cat > "$PKG/README.md" <<'MD'
# FOCUS paper — visual-first revision

Self-contained tarball of the paper + presentation + figures.

## Layout
- `paper_visual.tex` — IEEE-conference paper, figure-led revision.
- `presentation/slides.tex` — beamer 16:9 deck reusing the same PDFs.
- `report/sections/diagrams_focus.tex` — TikZ algorithmic diagrams.
- `report/sections/explainer_basics.tex` — basics→advanced sidebar.
- `results/figures/` — 12 generated PDF+PNG figures.
- `results/*.json` — real-data fixtures the figures are built from.
- `scripts/build_visuals.py` — regenerate figures from JSON.

## Build
```bash
# Figures
python3 scripts/build_visuals.py
# Paper (Overleaf works; or locally)
pdflatex paper_visual && bibtex paper_visual && pdflatex paper_visual && pdflatex paper_visual
# Presentation
cd presentation && pdflatex slides
```

## What changed vs. paper_fixed.tex
- Hero figure (FOCUS explainer, 6 panels, basic→advanced) on page 1.
- TikZ diagrams: pipeline overview, assigner internals, L1→L2 hierarchy,
  training algorithm. Replace prose paragraphs.
- 12 real-data figures dispersed through Results, not clustered.
- Pedagogical sidebar so a reader can stop reading at any depth.
- Page count down ~30% by replacing text with visuals (target ≥1/3 visual
  per page, often 2/3).
MD

# Zip.
( cd "$OUT_DIR" && zip -qr "focus_paper_visuals.zip" "focus_paper_visuals" )

echo
echo "  built: $ZIP"
echo "  size:  $(du -h "$ZIP" | cut -f1)"
echo "  files: $(unzip -l "$ZIP" | tail -1)"
