#!/usr/bin/env bash
# =====================================================================
# local_runner.sh  (LOCAL-GPU VARIANT - zero rental cost path)
# =====================================================================
# Same pipeline as the controller MINUS provisioning/teardown: it
# assumes a LOCAL CUDA GPU is already present. It runs bootstrap.sh +
# run_parallel.sh (Stages A-C incl. severe_tests.py), then
# audit_checklist_gen.py, then opens the SAME kind of human-gated
# results PR via `gh`.
#
# INTEGRITY GUARDRAIL (NON-NEGOTIABLE): infrastructure only. It NEVER
# edits journal-synthesis/main.tex, never writes a paper claim, never
# auto-merges. The PR is labelled needs-human-audit and carries the
# DO-NOT-MERGE banner (from AUDIT_CHECKLIST.md). A human signs off.
#
# SECRETS: gh uses its own auth (GH_TOKEN env / `gh auth login`). No
# token is embedded here. VAST_API_KEY is NOT needed (no rental).
#
# Usage:
#   export CHECKPOINT=/path/to/ckpt CORD=cord=/path/to/cord/test
#   bash cicd/local_runner.sh [--no-pr] [--branch NAME]
#
# Unattended local iteration (DELIBERATE opt-in only):
#   --- cron (every Monday 04:00) ---
#   0 4 * * 1  cd /path/kaggle2/journal-synthesis/vastai && \
#              bash cicd/local_runner.sh >> /var/log/journal-exp.log 2>&1
#
#   --- systemd timer ---
#   # /etc/systemd/system/journal-exp.service
#   [Unit]
#   Description=Journal experiments (local GPU, human-gated PR)
#   [Service]
#   Type=oneshot
#   WorkingDirectory=/path/kaggle2/journal-synthesis/vastai
#   ExecStart=/usr/bin/bash cicd/local_runner.sh
#   # /etc/systemd/system/journal-exp.timer
#   [Unit]
#   Description=Weekly journal experiments
#   [Timer]
#   OnCalendar=Mon *-*-* 04:00:00
#   Persistent=true
#   [Install]
#   WantedBy=timers.target
#   # then: systemctl enable --now journal-exp.timer
#   NOTE: enabling the timer/cron is UNATTENDED iteration. Do a manual
#   supervised run first; the PR ALWAYS needs a human before any number
#   reaches the paper.
# =====================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"     # .../cicd
VASTAI="$(cd "$HERE/.." && pwd)"                          # .../vastai
REPO_ROOT="$(cd "$VASTAI/../.." && pwd)"
MAIN_TEX="$REPO_ROOT/journal-synthesis/main.tex"

DO_PR=1
BRANCH="experiments/run-$(date -u +%Y%m%d-%H%M)"
while [ $# -gt 0 ]; do
  case "$1" in
    --no-pr)   DO_PR=0; shift ;;
    --branch)  BRANCH="${2:?}"; shift 2 ;;
    *) echo "[local] unknown arg: $1" >&2; exit 2 ;;
  esac
done

log() { echo "[local] $*"; }

# Integrity guard: snapshot main.tex hash; assert untouched at the end.
_tex_hash() { [ -f "$MAIN_TEX" ] && sha256sum "$MAIN_TEX" | awk '{print $1}' || echo "absent"; }
TEX_BEFORE="$(_tex_hash)"

bash "$HERE/notify.sh" start "local-GPU journal run starting" || true

cd "$VASTAI"
log "bootstrap.sh"
bash bootstrap.sh || { log "bootstrap failed"; bash "$HERE/notify.sh" failure "bootstrap failed"; exit 1; }

# shellcheck disable=SC1091
[ -f .env.sh ] && source .env.sh || true
: "${CHECKPOINT:?set CHECKPOINT (export or via .env.sh)}"
: "${CORD:?set CORD=label=/path}"

log "run_parallel.sh (Stages A-C incl. severe_tests.py)"
RC=0
bash run_parallel.sh || RC=$?

log "audit_checklist_gen.py (interprets; does NOT decide)"
python3 "$HERE/audit_checklist_gen.py" || true

TEX_AFTER="$(_tex_hash)"
if [ "$TEX_BEFORE" != "$TEX_AFTER" ]; then
  log "FATAL: main.tex changed during the run. This automation must"
  log "       NEVER touch the paper. Refusing to open a PR."
  bash "$HERE/notify.sh" failure "main.tex changed - aborted" || true
  exit 1
fi

if [ "$RC" -ne 0 ]; then
  log "pipeline reported failures (rc=$RC); results still fetched."
  bash "$HERE/notify.sh" failure "pipeline rc=$RC (results present)" || true
fi

if [ "$DO_PR" -ne 1 ]; then
  log "--no-pr: skipping PR. Raw results in $VASTAI/results/."
  log "A HUMAN must audit vs PREREGISTRATION.md before any paper edit."
  exit "$RC"
fi

if ! command -v gh >/dev/null 2>&1; then
  log "gh not installed -> cannot open PR. Raw results are in"
  log "$VASTAI/results/ ; open the human-gated PR manually."
  exit "$RC"
fi

cd "$REPO_ROOT"
RES="journal-synthesis/vastai/results"
log "opening human-gated results PR on branch $BRANCH"
git checkout -b "$BRANCH" 2>/dev/null || git checkout "$BRANCH"
# Stage ONLY results/checklist - NEVER main.tex.
git add "$RES"/*.json "$RES"/SEVERE.json "$RES"/AUDIT_CHECKLIST.md 2>/dev/null || true
if ! git diff --cached --quiet -- "journal-synthesis/main.tex"; then
  log "FATAL: main.tex is staged. Refusing. (This must never happen.)"
  exit 1
fi
git -c user.email="local-runner@infra" -c user.name="journal-local-runner" \
  commit -m "experiments: raw results + audit checklist (infra only, human-gated, no paper write)" \
  || { log "nothing to commit"; exit "$RC"; }
git push -u origin "$BRANCH" || { log "push failed"; exit 1; }
gh pr create \
  --title "Experiment run results (NEEDS HUMAN AUDIT - DO NOT MERGE INTO PAPER)" \
  --body-file "$RES/AUDIT_CHECKLIST.md" \
  --label needs-human-audit \
  --draft \
  || log "gh pr create failed (open the PR manually; do NOT auto-merge)"

bash "$HERE/notify.sh" success "local run done; human-gated PR opened" || true
log "DONE. The PR is needs-human-audit + DO-NOT-MERGE. Human signs off."
exit "$RC"
