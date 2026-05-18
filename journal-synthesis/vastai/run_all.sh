#!/usr/bin/env bash
# Ordered driver for the vast.ai journal experiment package.
#
# Fail-fast. Each experiment echoes which one is running and appends a
# line to results/MANIFEST.txt on success. The CHECKPOINT and corpus
# paths MUST be supplied by the caller (no model id is baked in). Edit
# the variables below or export them before running. See
# README_RUNBOOK.md for how to fetch models/data first.
#
# This script does NOT fabricate anything: if a step fails (e.g. no GPU)
# it aborts and the manifest shows exactly how far it got.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

: "${CHECKPOINT:?set CHECKPOINT to the KIE checkpoint path/id}"
: "${CORD:?set CORD=cord=/path/to/cord (label=path)}"
SROIE="${SROIE:-}"          # optional, label=path
WILDRECEIPT="${WILDRECEIPT:-}"  # optional, label=path
TASK_PROMPT="${TASK_PROMPT:-<s_cord-v2>}"

mkdir -p results
MANIFEST="results/MANIFEST.txt"
: > "$MANIFEST"
echo "run_all started $(date -u +%FT%TZ)" >> "$MANIFEST"

CORPORA=("$CORD")
[ -n "$SROIE" ] && CORPORA+=("$SROIE")
[ -n "$WILDRECEIPT" ] && CORPORA+=("$WILDRECEIPT")

step() {  # name, then full command
  local name="$1"; shift
  echo "=================================================================="
  echo ">>> RUNNING $name"
  echo "=================================================================="
  "$@"
  echo "$name OK $(date -u +%FT%TZ)" >> "$MANIFEST"
}

step "E1E3_fullscale" python3 e1e3_fullscale.py \
  --checkpoint "$CHECKPOINT" --task_prompt "$TASK_PROMPT" --corpus "$CORD"

step "E5_integrated_benchmark" python3 e5_integrated_benchmark.py \
  --checkpoint "$CHECKPOINT" --task_prompt "$TASK_PROMPT" \
  --corpora "${CORPORA[@]}"

if [ -n "$SROIE" ] || [ -n "$WILDRECEIPT" ]; then
  PAIRS=()
  [ -n "$SROIE" ] && PAIRS+=("${CORD%%=*}:${SROIE%%=*}")
  [ -n "$WILDRECEIPT" ] && PAIRS+=("${CORD%%=*}:${WILDRECEIPT%%=*}")
  step "E6_multi_shift_pairs" python3 e6_multi_shift_pairs.py \
    --checkpoint "$CHECKPOINT" --task_prompt "$TASK_PROMPT" \
    --corpora "${CORPORA[@]}" --pairs "${PAIRS[@]}"
else
  echo "E6 SKIPPED (no shift corpus set) $(date -u +%FT%TZ)" >> "$MANIFEST"
fi

step "E7_mechanism_synthetic_shift" python3 e7_mechanism_synthetic_shift.py \
  --checkpoint "$CHECKPOINT" --task_prompt "$TASK_PROMPT" --base "$CORD"

step "E8_end_to_end_latency" python3 e8_end_to_end_latency.py \
  --checkpoint "$CHECKPOINT" --task_prompt "$TASK_PROMPT" --corpus "$CORD"

step "E9_alt_verifier_bakeoff" python3 e9_alt_verifier_bakeoff.py \
  --checkpoint "$CHECKPOINT" --task_prompt "$TASK_PROMPT" --corpus "$CORD"

step "E10_power_and_breadth" python3 e10_power_and_breadth.py \
  --checkpoints "$CHECKPOINT" --task_prompt "$TASK_PROMPT" \
  --corpora "${CORPORA[@]}"

echo "run_all finished $(date -u +%FT%TZ)" >> "$MANIFEST"
echo "ALL DONE. See results/ and results/MANIFEST.txt"
