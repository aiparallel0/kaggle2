#!/usr/bin/env bash
# =====================================================================
# teardown.sh  (PHASE 0 - the real cost saver)
# =====================================================================
# Destroys the current vast.ai instance. INFRASTRUCTURE ONLY: it touches
# no results, no paper, no science. It is the brake that stops a forgot-
# ten rented GPU from burning money.
#
# Two modes:
#   teardown.sh [INSTANCE_ID]
#       Destroy the given instance (or the vast.ai-provided
#       $CONTAINER_ID / $VAST_CONTAINERLABEL-derived id / $INSTANCE_ID).
#       Idempotent: destroying an already-gone instance is OK (vast_api
#       swallows 404). Safe to call repeatedly (trap EXIT, retries...).
#
#   teardown.sh --cost-cap-watchdog SECONDS [INSTANCE_ID]
#       Sleep SECONDS then FORCE-destroy. This is the wall-clock cost
#       cap: even if every other safety fails, the box dies after the
#       cap. Run this in the background from the runner.
#
# SECRETS: never inline. The API key is read by vast_api.py from
# $VAST_API_KEY only. Nothing here echoes it.
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PYTHON:-python3}"
VAST_API="$HERE/vast_api.py"

log() { echo "[teardown] $*" >&2; }

resolve_id() {  # echo the instance id from arg/env, or empty
  local arg="${1:-}"
  if [ -n "$arg" ]; then echo "$arg"; return; fi
  # vast.ai injects these into the container; try the common ones.
  if [ -n "${INSTANCE_ID:-}" ];     then echo "$INSTANCE_ID"; return; fi
  if [ -n "${VAST_INSTANCE_ID:-}" ]; then echo "$VAST_INSTANCE_ID"; return; fi
  if [ -n "${CONTAINER_ID:-}" ];    then echo "$CONTAINER_ID"; return; fi
  echo ""
}

destroy_once() {
  local id="$1"
  if [ -z "$id" ]; then
    log "no instance id (arg/env) -> nothing to destroy (no-op, OK)."
    return 0
  fi
  if [ -z "${VAST_API_KEY:-}" ]; then
    log "VAST_API_KEY unset -> cannot call vast.ai. NOT failing the"
    log "run for this (the box may be a local/dev box). No-op."
    return 0
  fi
  local attempt=1 max=4 wait=2
  while [ "$attempt" -le "$max" ]; do
    if "$PY" "$VAST_API" destroy --instance-id "$id"; then
      log "instance $id destroy request accepted (attempt $attempt)."
      return 0
    fi
    log "destroy attempt $attempt failed; retry in ${wait}s"
    sleep "$wait"; wait=$((wait * 2)); attempt=$((attempt + 1))
  done
  log "ERROR: could not destroy instance $id after $max attempts."
  log "       MANUALLY destroy it in the vast.ai console NOW."
  return 1
}

main() {
  if [ "${1:-}" = "--cost-cap-watchdog" ]; then
    local secs="${2:-}"; local id
    id="$(resolve_id "${3:-}")"
    if ! [[ "$secs" =~ ^[0-9]+$ ]] || [ "$secs" -le 0 ]; then
      log "--cost-cap-watchdog needs a positive integer SECONDS"; exit 2
    fi
    log "COST-CAP WATCHDOG armed: force-destroy instance '${id:-?}' in ${secs}s"
    sleep "$secs"
    log "COST CAP REACHED (${secs}s elapsed) -> force destroying ${id:-?}"
    destroy_once "$id"
    exit $?
  fi
  local id; id="$(resolve_id "${1:-}")"
  destroy_once "$id"
  exit $?
}

main "$@"
