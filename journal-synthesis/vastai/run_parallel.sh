#!/usr/bin/env bash
# =====================================================================
# GPU-AWARE PARALLEL + RESUMABLE RUNNER
# =====================================================================
# Optimisation that is HONEST about its limits:
#  - Every experiment here is GPU-bound (each loads the KIE model and
#    decodes). On a SINGLE GPU true parallelism gives little speedup;
#    the real wins on 1 GPU are RESUMABILITY and per-job timing.
#  - On N GPUs this shards up to N experiments concurrently (one GPU
#    each via CUDA_VISIBLE_DEVICES) -> near-linear speedup up to N.
#  - It does NOT pretend to share inference across experiments: the
#    packaged scripts are self-contained and each re-decodes. A shared
#    decode cache would need a scripts refactor; that is documented as
#    a future optimisation, not silently claimed here.
#
# Resumable: an experiment whose results/<EXP>.json already carries a
# real `computed_on` stamp is SKIPPED (never recomputed, never faked).
# Fail-soft: a failed job is recorded; other jobs continue; the script
# exits non-zero if any job failed so you cannot miss it.
#
#   source .env.sh && bash run_parallel.sh
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$HERE"
: "${CHECKPOINT:?set CHECKPOINT (e.g. source .env.sh)}"
: "${CORD:?set CORD=label=/path}"
SROIE="${SROIE:-}"; WILDRECEIPT="${WILDRECEIPT:-}"
TASK_PROMPT="${TASK_PROMPT:-<s_cord-v2>}"
mkdir -p results
TIMING="results/PARALLEL_TIMING.tsv"
[ -f "$TIMING" ] || printf "exp\tstart_utc\tend_utc\tseconds\texit\n" > "$TIMING"

NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
[ "$NGPU" -lt 1 ] && { echo "FATAL: no GPU visible (nvidia-smi)"; exit 1; }
echo "[parallel] detected $NGPU GPU(s)"

CORPORA=("$CORD"); [ -n "$SROIE" ] && CORPORA+=("$SROIE")
[ -n "$WILDRECEIPT" ] && CORPORA+=("$WILDRECEIPT")
PAIRS=(); [ -n "$SROIE" ] && PAIRS+=("${CORD%%=*}:${SROIE%%=*}")
[ -n "$WILDRECEIPT" ] && PAIRS+=("${CORD%%=*}:${WILDRECEIPT%%=*}")

# Job table: NAME|||command (heaviest first so a free GPU never idles).
JOBS=()
JOBS+=("E5_integrated_benchmark|||python3 e5_integrated_benchmark.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpora ${CORPORA[*]}")
JOBS+=("E7_mechanism_synthetic_shift|||python3 e7_mechanism_synthetic_shift.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --base $CORD")
JOBS+=("E1E3_fullscale|||python3 e1e3_fullscale.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpus $CORD")
if [ "${#PAIRS[@]}" -gt 0 ]; then
  JOBS+=("E6_multi_shift_pairs|||python3 e6_multi_shift_pairs.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpora ${CORPORA[*]} --pairs ${PAIRS[*]}")
fi
JOBS+=("E9_alt_verifier_bakeoff|||python3 e9_alt_verifier_bakeoff.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpus $CORD")
JOBS+=("E10_power_and_breadth|||python3 e10_power_and_breadth.py --checkpoints $CHECKPOINT --task_prompt $TASK_PROMPT --corpora ${CORPORA[*]}")
JOBS+=("E8_end_to_end_latency|||python3 e8_end_to_end_latency.py --checkpoint $CHECKPOINT --task_prompt $TASK_PROMPT --corpus $CORD")

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

run_one() {  # gpu_idx  NAME  CMD...
  local g="$1" name="$2"; shift 2
  local log="results/${name}.log" s e rc
  s="$(date -u +%FT%TZ)"
  echo "[parallel] >>> $name on GPU $g"
  CUDA_VISIBLE_DEVICES="$g" "$@" >"$log" 2>&1
  rc=$?
  e="$(date -u +%FT%TZ)"
  local secs=$(( $(date -u -d "$e" +%s) - $(date -u -d "$s" +%s) ))
  printf "%s\t%s\t%s\t%s\t%s\n" "$name" "$s" "$e" "$secs" "$rc" >> "$TIMING"
  [ $rc -eq 0 ] && echo "[parallel] OK  $name (${secs}s)" \
                || echo "[parallel] FAIL $name rc=$rc (see $log)"
  return $rc
}

declare -a GPU_PID   # pid occupying each gpu slot
for ((i=0;i<NGPU;i++)); do GPU_PID[$i]=0; done
FAILED=0

for entry in "${JOBS[@]}"; do
  name="${entry%%|||*}"; cmd="${entry##*|||}"
  if already_done "$name"; then
    echo "[parallel] SKIP $name (real result already present)"
    continue
  fi
  # wait for a free GPU slot
  placed=0
  while [ $placed -eq 0 ]; do
    for ((i=0;i<NGPU;i++)); do
      pid="${GPU_PID[$i]}"
      if [ "$pid" -eq 0 ] || ! kill -0 "$pid" 2>/dev/null; then
        if [ "$pid" -ne 0 ]; then wait "$pid" || FAILED=1; fi
        run_one "$i" "$name" $cmd &
        GPU_PID[$i]=$!
        placed=1; break
      fi
    done
    [ $placed -eq 0 ] && sleep 5
  done
done

for ((i=0;i<NGPU;i++)); do
  pid="${GPU_PID[$i]}"
  [ "$pid" -ne 0 ] && { wait "$pid" || FAILED=1; }
done

echo "=================================================================="
column -t -s $'\t' "$TIMING" 2>/dev/null || cat "$TIMING"
echo "=================================================================="
if [ $FAILED -ne 0 ]; then
  echo "[parallel] SOME JOBS FAILED. Inspect results/<EXP>.log. Re-run"
  echo "           this script to resume (completed ones are skipped)."
  exit 1
fi
echo "[parallel] ALL JOBS DONE. Audit results/ against PREREGISTRATION.md"
echo "           before any number is moved into the paper (internal QA)."
