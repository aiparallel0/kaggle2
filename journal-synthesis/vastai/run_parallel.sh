#!/usr/bin/env bash
# =====================================================================
# DECODE-ONCE + RESUMABLE RUNNER (honest about its single-GPU limits)
# =====================================================================
# THE COST FIX. Previously every experiment independently loaded the KIE
# model and RE-DECODED the same corpora, so one run paid for the SAME
# Donut inference ~7x (~90% wasted spend on a paid GPU). Now:
#
#  - STAGE A (GPU, ONE pass): for each DISTINCT corpus the shared decode
#    (common/records.decode_or_load) runs exactly ONCE, writing
#    results/<label>__<ckpthash>.records.jsonl. This is the only GPU
#    work for the cache-consuming experiments.
#  - STAGE B (CPU, parallel): E1E3 / E5 / E6 / E9 / E10 then read that
#    cache (NO model load, NO GPU) and only do their analysis math, so
#    they are CPU-bound and run concurrently up to nproc.
#  - E7 (synthetic per-cell perturbed decodes) and E8 (per-receipt
#    latency MEASUREMENT) genuinely need the GPU and are NOT cache
#    consumers; they run on the GPU after Stage A. Their decode is NOT
#    the redundant re-decode the cache removes, so it is honest to keep
#    them inline.
#
# This removes the ~7x redundant decode. It does NOT claim multi-GPU
# scaling: there is one GPU; the win is decode-once + CPU-parallel
# analyses + resumability, not GPU sharding.
#
# Resumable: an experiment whose results/<EXP>.json already carries a
# real `computed_on` stamp is SKIPPED. A complete Stage-A cache file is
# likewise reused (decode_or_load checks header + record count and
# rebuilds a truncated cache, never half-uses it).
# Fail-soft: a failed job is recorded; others continue; non-zero exit
# if any job failed so you cannot miss it.
#
#   source .env.sh && BATCH=16 bash run_parallel.sh
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"

# =====================================================================
# OPT-IN SELF-DESTROY / COST-CAP  (CI/CD PHASE 0 wiring)
# =====================================================================
# DEFAULT: OFF. This block does NOTHING unless SELF_DESTROY=1 is set
# AND a vast.ai instance id AND VAST_API_KEY are present. That gate is
# deliberate: a developer running this on a local/dev box must NEVER
# have the machine destroyed by surprise. When (and only when) every
# condition holds, an EXIT trap tears the rented instance down on
# completion / failure / signal, and an independent background
# wall-clock killer force-destroys after MAX_RUN_MINUTES (also unset =
# OFF) as a last-resort cost cap. INFRASTRUCTURE ONLY: teardown.sh
# touches no results and no paper.
SELF_DESTROY="${SELF_DESTROY:-0}"
MAX_RUN_MINUTES="${MAX_RUN_MINUTES:-}"   # unset/empty = OFF (no killer)
_CICD="$HERE/cicd"
_SD_ID=""
for _v in "${SELF_DESTROY_INSTANCE_ID:-}" "${INSTANCE_ID:-}" \
          "${VAST_INSTANCE_ID:-}" "${CONTAINER_ID:-}"; do
  [ -n "$_v" ] && { _SD_ID="$_v"; break; }
done
_SD_WATCHDOG_PID=""
if [ "$SELF_DESTROY" = "1" ] && [ -n "$_SD_ID" ] \
   && [ -n "${VAST_API_KEY:-}" ] && [ -f "$_CICD/teardown.sh" ]; then
  echo "[parallel] SELF_DESTROY armed for instance $_SD_ID "\
"(teardown on EXIT; cost cap=${MAX_RUN_MINUTES:-OFF} min)"
  _self_destroy() {
    local rc=$?
    echo "[parallel] EXIT trap -> tearing down instance $_SD_ID (rc=$rc)"
    [ -n "$_SD_WATCHDOG_PID" ] && kill "$_SD_WATCHDOG_PID" 2>/dev/null || true
    bash "$_CICD/teardown.sh" "$_SD_ID" || true
  }
  trap _self_destroy EXIT INT TERM
  if [ -n "$MAX_RUN_MINUTES" ] && [[ "$MAX_RUN_MINUTES" =~ ^[0-9]+$ ]] \
     && [ "$MAX_RUN_MINUTES" -gt 0 ]; then
    _CAP_SECS=$((MAX_RUN_MINUTES * 60))
    bash "$_CICD/teardown.sh" --cost-cap-watchdog "$_CAP_SECS" "$_SD_ID" &
    _SD_WATCHDOG_PID=$!
    echo "[parallel] background cost-cap killer pid=$_SD_WATCHDOG_PID "\
"(force-destroy in ${_CAP_SECS}s)"
  fi
else
  [ "$SELF_DESTROY" = "1" ] && echo "[parallel] SELF_DESTROY=1 but "\
"instance id / VAST_API_KEY / teardown.sh missing -> NOT arming "\
"self-destroy (safe default; dev box protected)."
fi
# =====================================================================

: "${CHECKPOINT:?set CHECKPOINT (e.g. source .env.sh)}"
: "${CORD:?set CORD=label=/path}"
SROIE="${SROIE:-}"; WILDRECEIPT="${WILDRECEIPT:-}"
TASK_PROMPT="${TASK_PROMPT:-<s_cord-v2>}"
BATCH="${BATCH:-16}"
mkdir -p results
TIMING="results/PARALLEL_TIMING.tsv"
[ -f "$TIMING" ] || printf "exp\tstart_utc\tend_utc\tseconds\texit\n" > "$TIMING"

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
[ "$NGPU" -lt 1 ] && { echo "FATAL: no GPU visible (nvidia-smi)"; exit 1; }
NPROC="$(nproc 2>/dev/null || echo 2)"
echo "[parallel] detected $NGPU GPU(s), $NPROC CPU(s), BATCH=$BATCH"

CORPORA=("$CORD"); [ -n "$SROIE" ] && CORPORA+=("$SROIE")
[ -n "$WILDRECEIPT" ] && CORPORA+=("$WILDRECEIPT")
PAIRS=(); [ -n "$SROIE" ] && PAIRS+=("${CORD%%=*}:${SROIE%%=*}")
[ -n "$WILDRECEIPT" ] && PAIRS+=("${CORD%%=*}:${WILDRECEIPT%%=*}")

FAILED=0

# ---------------------------------------------------------------------
# STAGE A: decode each DISTINCT corpus ONCE (the single GPU pass for the
# cache-consuming experiments). Idempotent: decode_or_load reuses a
# complete cache file and rebuilds a truncated one.
# ---------------------------------------------------------------------
echo "[parallel] === STAGE A: decode-once shared cache ==="
declare -A SEEN_CORPUS
for lp in "${CORPORA[@]}"; do
  [ -n "${SEEN_CORPUS[$lp]:-}" ] && continue
  SEEN_CORPUS[$lp]=1
  label="${lp%%=*}"
  s="$(date -u +%FT%TZ)"
  echo "[parallel] >>> Stage-A decode $label"
  python3 - "$lp" "$CHECKPOINT" "$TASK_PROMPT" "$BATCH" \
      >"results/StageA_${label}.log" 2>&1 <<'PY'
import sys
sys.path.insert(0, ".")
from common import decode_or_load
corpus_arg, ckpt, prompt, batch = sys.argv[1:5]
recs = decode_or_load(corpus_arg, ckpt, prompt, int(batch))
n = len(recs)
print("decoded/loaded %d receipts for %s" % (n, corpus_arg))
# Hard sanity stamp: a Stage-A decode/load that yields 0 records is a
# FAIL, never a "complete" cache downstream silently consumes (this is
# exactly the WildReceipt n_records:0 collapse). decode_or_load already
# raises on a genuine 0-record corpus / refuses a 0-record cache; this
# is the belt-and-braces exit code in case a loader ever returns [].
if n == 0:
    sys.stderr.write(
        "FAIL: Stage-A produced 0 records for %s. Refusing to treat "
        "this as a complete cache; fix the corpus layout / loader.\n"
        % corpus_arg)
    sys.exit(3)
PY
  rc=$?
  e="$(date -u +%FT%TZ)"
  secs=$(( $(date -u -d "$e" +%s) - $(date -u -d "$s" +%s) ))
  printf "%s\t%s\t%s\t%s\t%s\n" "StageA_${label}" "$s" "$e" "$secs" "$rc" \
      >> "$TIMING"
  if [ $rc -eq 0 ]; then
    echo "[parallel] OK  Stage-A $label (${secs}s)"
  else
    echo "[parallel] FAIL Stage-A $label rc=$rc (see results/StageA_${label}.log)"
    FAILED=1
  fi
done

already_done() {  # EXP -> 0 if a real result exists
  python3 - "$1" <<'PY' 2>/dev/null
import json,sys,glob,os
exp=sys.argv[1]
for f in glob.glob(os.path.join("results",exp+".json")):
    try:
        d=json.load(open(f))
    except Exception:
        sys.exit(1)
    if isinstance(d,dict) and d.get("computed_on"):
        sys.exit(0)
sys.exit(1)
PY
}

run_one() {  # slot  NAME  CMD...   slot="cpuN" -> no GPU; integer -> that GPU
  local g="$1" name="$2"; shift 2
  local log="results/${name}.log" s e rc secs vis
  s="$(date -u +%FT%TZ)"
  case "$g" in
    cpu*) vis=""  ; echo "[parallel] >>> $name (CPU slot $g, no GPU)" ;;
    *)    vis="$g"; echo "[parallel] >>> $name (GPU $g)" ;;
  esac
  CUDA_VISIBLE_DEVICES="$vis" "$@" >"$log" 2>&1
  rc=$?
  e="$(date -u +%FT%TZ)"
  secs=$(( $(date -u -d "$e" +%s) - $(date -u -d "$s" +%s) ))
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$s" "$e" "$secs" "$rc" >> "$TIMING"
  [ $rc -eq 0 ] && echo "[parallel] OK  $name (${secs}s)" \
                || echo "[parallel] FAIL $name rc=$rc (see $log)"
  return $rc
}

# ---------------------------------------------------------------------
# STAGE B-cpu: cache-consuming, CPU-bound. They no longer touch the GPU
# (Stage A produced the cache), so run them concurrently up to nproc.
# ---------------------------------------------------------------------
CPU_JOBS=()
CPU_JOBS+=("E1E3_fullscale|||python3 e1e3_fullscale.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpus $CORD --batch $BATCH")
CPU_JOBS+=("E5_integrated_benchmark|||python3 e5_integrated_benchmark.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpora ${CORPORA[*]} --batch $BATCH")
if [ "${#PAIRS[@]}" -gt 0 ]; then
  CPU_JOBS+=("E6_multi_shift_pairs|||python3 e6_multi_shift_pairs.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpora ${CORPORA[*]} --pairs ${PAIRS[*]} --batch $BATCH")
fi
CPU_JOBS+=("E9_alt_verifier_bakeoff|||python3 e9_alt_verifier_bakeoff.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpus $CORD --batch $BATCH")
CPU_JOBS+=("E10_power_and_breadth|||python3 e10_power_and_breadth.py --checkpoints $CHECKPOINT --task_prompt $TASK_PROMPT --corpora ${CORPORA[*]} --batch $BATCH")

echo "[parallel] === STAGE B (CPU): cache-consuming analyses, up to $NPROC concurrent ==="
declare -a SLOT_PID
for ((i=0;i<NPROC;i++)); do SLOT_PID[$i]=0; done

for entry in "${CPU_JOBS[@]}"; do
  name="${entry%%|||*}"; cmd="${entry##*|||}"
  if already_done "$name"; then
    echo "[parallel] SKIP $name (real result already present)"
    continue
  fi
  placed=0
  while [ $placed -eq 0 ]; do
    for ((i=0;i<NPROC;i++)); do
      pid="${SLOT_PID[$i]}"
      if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
        if [ "$pid" -ne 0 ]; then wait "$pid" || FAILED=1; fi
        # CPU stage: no GPU; pass slot index only for logging.
        run_one "cpu${i}" "$name" $cmd &
        SLOT_PID[$i]=$!
        placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 5
  done
done
for ((i=0;i<NPROC;i++)); do
  pid="${SLOT_PID[$i]}"
  [ "$pid" -ne 0 ] && { wait "$pid" || FAILED=1; }
done

# ---------------------------------------------------------------------
# STAGE B-gpu: genuinely GPU-bound, NOT cache consumers (E7 perturbed
# per-cell decodes, E8 per-receipt latency MEASUREMENT). One GPU -> run
# them sequentially after the CPU stage so they get the full GPU.
# ---------------------------------------------------------------------
GPU_JOBS=()
GPU_JOBS+=("E7_mechanism_synthetic_shift|||python3 e7_mechanism_synthetic_shift.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --base $CORD --batch $BATCH")
GPU_JOBS+=("E8_end_to_end_latency|||python3 e8_end_to_end_latency.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpus $CORD --batch $BATCH")

echo "[parallel] === STAGE B (GPU): genuine GPU work, sequential on 1 GPU ==="
for entry in "${GPU_JOBS[@]}"; do
  name="${entry%%|||*}"; cmd="${entry##*|||}"
  if already_done "$name"; then
    echo "[parallel] SKIP $name (real result already present)"
    continue
  fi
  run_one 0 "$name" $cmd || FAILED=1
done

# ---------------------------------------------------------------------
# STAGE C (CPU): SEVERE TESTS S1-S4. Robustness-checks the NEGATIVE
# result; NOT a thesis-rescue. Cache-only (reuses the Stage-A
# decode-once cache via the same cache-load path), NO GPU, NO decode,
# single fast job (~seconds). Resumable: a SEVERE.json that already
# carries a real `computed_on` is skipped by already_done() exactly
# like the Stage-B experiments. Runs AFTER Stage B so the caches it
# reads are guaranteed present.
# ---------------------------------------------------------------------
echo "[parallel] === STAGE C (CPU): severe tests S1-S4 (cache-only, no GPU) ==="
if already_done "SEVERE"; then
  echo "[parallel] SKIP SEVERE (real result already present)"
else
  run_one "cpu0" "SEVERE" python3 severe_tests.py \
      --checkpoint "$CHECKPOINT" --task_prompt "$TASK_PROMPT" \
      --corpora "${CORPORA[@]}" || FAILED=1
fi

echo "=================================================================="
column -t -s $'\t' "$TIMING" 2>/dev/null || cat "$TIMING"
echo "=================================================================="
if [ $FAILED -ne 0 ]; then
  echo "[parallel] SOME JOBS FAILED. Inspect results/<EXP>.log /"
  echo "           results/StageA_<corpus>.log. Re-run this script to"
  echo "           resume (done experiments + complete caches are reused)."
  exit 1
fi
echo "[parallel] ALL JOBS DONE. Audit results/ against PREREGISTRATION.md"
echo "           before any number is moved into the paper (internal QA)."
