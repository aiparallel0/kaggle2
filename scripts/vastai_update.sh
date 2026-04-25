#!/usr/bin/env bash
# kaggle2/scripts/vastai_update.sh
# Pull the latest changes from GitHub onto an already-running vast.ai instance
# WITHOUT wiping the workspace or re-cloning.
#
# Usage (from inside the cloned repo on the vast.ai instance):
#   bash scripts/vastai_update.sh
#
# What it does:
#   1. Sanity-checks you are inside the kaggle2 repo.
#   2. git pull --rebase (fast-forward; aborts cleanly if you have local edits).
#   3. pip install -r requirements.txt  (picks up any new/changed deps).
#   4. make check  (mypy --strict + ruff — the kaggle2 test suite).
#
# After it completes, run:
#   make all        — full train + eval + paper
#   make train      — training only
#   make eval       — evaluation only (needs a completed run)
#   make paper      — paper generation only
#
# Tip: if you have local edits you want to keep, stash them first:
#   git stash && bash scripts/vastai_update.sh && git stash pop
set -euo pipefail
if [ "${VERBOSE:-0}" = "1" ]; then set -x; fi

log() { printf "\033[1;36m[update]\033[0m %s\n" "$*"; }
err() { printf "\033[1;31m[update ERROR]\033[0m %s\n" "$*" >&2; }

# ── 1. Sanity check ──────────────────────────────────────────────────────────
if [ ! -f "AGENTS.md" ] || [ ! -f "main.py" ]; then
    err "Run this script from the root of the kaggle2 repo."
    err "  cd /workspace/kaggle2 && bash scripts/vastai_update.sh"
    exit 1
fi

REMOTE_URL="$(git remote get-url origin 2>/dev/null || true)"
if [[ "$REMOTE_URL" != *"kaggle2"* ]]; then
    err "Remote origin ('$REMOTE_URL') does not look like the kaggle2 repo."
    exit 1
fi

# ── 2. Warn about local edits (non-fatal) ────────────────────────────────────
if ! git diff --quiet || ! git diff --cached --quiet; then
    log "WARNING: you have uncommitted local changes."
    log "  Stash them first if you don't want them to block the pull:"
    log "    git stash && bash scripts/vastai_update.sh && git stash pop"
fi

# ── 3. Pull ───────────────────────────────────────────────────────────────────
log "Fetching from origin…"
git fetch origin

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
log "Current branch: $BRANCH"

BEFORE="$(git rev-parse HEAD)"
git pull --rebase origin "$BRANCH"
AFTER="$(git rev-parse HEAD)"

if [ "$BEFORE" = "$AFTER" ]; then
    log "Already up to date ($(git rev-parse --short HEAD))."
else
    log "Updated $(git rev-parse --short "$BEFORE") → $(git rev-parse --short "$AFTER")"
    git log --oneline "${BEFORE}..${AFTER}"
fi

# ── 4. Sync Python deps (only if requirements.txt changed) ───────────────────
if ! git diff --quiet "${BEFORE}..${AFTER}" -- requirements.txt 2>/dev/null \
        || [ "$BEFORE" = "$AFTER" ]; then
    # Always re-run on first call even when already up to date so a
    # partially-installed environment is healed.
    log "Installing / syncing Python requirements…"
    pip install --upgrade pip -q
    pip install -r requirements.txt -q
    log "Requirements synced."
else
    log "requirements.txt unchanged — skipping pip install."
fi

# ── 5. Static checks ─────────────────────────────────────────────────────────
log "Running static checks (mypy --strict + ruff)…"
make check

log ""
log "Update complete. Ready to run:"
log "  make all        — full pipeline (train + eval + paper)"
log "  make train      — training only"
log "  make eval       — evaluation only"
log "  make paper      — paper generation only"