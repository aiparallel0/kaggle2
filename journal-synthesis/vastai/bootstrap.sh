#!/usr/bin/env bash
# =====================================================================
# FRESH vast.ai INSTANCE BOOTSTRAP  (idempotent; corrected after a live
# run exposed: private repos need a token, the prior fetch scripts take
# NO --out flag and write inside the arith-gating repo, and the CKPT
# placeholder must be a real id with no angle brackets).
# =====================================================================
# HONESTY / SCOPE:
#  - No torch via pip (vast.ai image ships a matched torch+torchvision).
#  - KIE checkpoint id is NOT baked in; you pass CKPT_ID (no < > ).
#  - Data is fetched ONLY by the prior repos' own fetch scripts, with
#    their REAL signatures; output dirs are then discovered, not guessed.
#
# REQUIRED env (set BEFORE running; replace placeholders, NO angle
# brackets, each export on its OWN line):
#   export GITHUB_TOKEN=ghp_xxx        # PAT with read access (private repos)
#   export CKPT_ID=naver-clova-ix/...  # a real HF Donut CORD-v2 ckpt id
#   export REPO_URL=https://github.com/aiparallel0/kaggle2.git
#   export ARITH_URL=https://github.com/aiparallel0/arith-gating.git
#   export TRIOLOGY_URL=https://github.com/aiparallel0/triology.git
#   bash bootstrap.sh
# =====================================================================
set -uo pipefail

BRANCH="${BRANCH:-claude/prepare-papers-repos-4LUdJ}"
WORK="${WORK:-/workspace}"
REPO_URL="${REPO_URL:?set REPO_URL (kaggle2 https remote)}"
ARITH_URL="${ARITH_URL:?set ARITH_URL (arith-gating https remote) - REQUIRED for data fetchers}"
TRIOLOGY_URL="${TRIOLOGY_URL:-}"      # optional: logic is lifted into pipeline.py
GITHUB_TOKEN="${GITHUB_TOKEN:-}"
CKPT_ID="${CKPT_ID:-}"
export HF_HOME="${HF_HOME:-$WORK/hf}"
export PYTHONUNBUFFERED=1
mkdir -p "$WORK" "$HF_HOME"

# Inject token for private https github clones (x-access-token works for
# both classic and fine-grained PATs).
_auth() {  # url -> tokenised url
  local u="$1"
  if [ -n "$GITHUB_TOKEN" ] && printf '%s' "$u" | grep -q '^https://github.com/'; then
    printf 'https://x-access-token:%s@github.com/%s' "$GITHUB_TOKEN" "${u#https://github.com/}"
  else
    printf '%s' "$u"
  fi
}
clone_or_pull() {  # url dir required(0/1)
  local url dir req; url="$(_auth "$1")"; dir="$2"; req="${3:-1}"
  if [ -d "$dir/.git" ]; then
    echo "[bootstrap] updating $dir"
    git -C "$dir" remote set-url origin "$url" 2>/dev/null || true
    git -C "$dir" fetch -q origin "$BRANCH" 2>/dev/null \
      && git -C "$dir" checkout -q "$BRANCH" 2>/dev/null \
      && git -C "$dir" pull -q --rebase origin "$BRANCH" 2>/dev/null || true
  else
    echo "[bootstrap] cloning $2 -> $dir"
    if ! GIT_TERMINAL_PROMPT=0 git clone -q --branch "$BRANCH" --single-branch "$url" "$dir" 2>/dev/null \
       && ! GIT_TERMINAL_PROMPT=0 git clone -q "$url" "$dir" 2>/dev/null; then
      if [ "$req" -eq 1 ]; then
        echo "FATAL: could not clone $2 (private?). Set GITHUB_TOKEN to a PAT with read access and re-run." >&2
        exit 1
      fi
      echo "[bootstrap] WARN: optional repo $2 not cloned (not required at run time)."
    fi
  fi
}

echo "== 1. system deps =="
( sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip tesseract-ocr ) \
  || echo "[bootstrap] apt step non-fatal (likely already present)"

echo "== 2. repos =="
clone_or_pull "$REPO_URL"  "$WORK/kaggle2"        1
clone_or_pull "$ARITH_URL" "$WORK/arith-gating"   1
[ -n "$TRIOLOGY_URL" ] && clone_or_pull "$TRIOLOGY_URL" "$WORK/triology" 0
PKG="$WORK/kaggle2/journal-synthesis/vastai"
AG="$WORK/arith-gating"

echo "== 3. python deps (NOT torch) =="
python3 -m pip install -q --upgrade --root-user-action=ignore pip
python3 -m pip install -q --root-user-action=ignore -r "$PKG/requirements.txt"

echo "== 4. GPU sanity =="
python3 - <<'PY' || exit 1
import sys
try: import torch
except Exception as e: sys.exit(f"FATAL: torch not importable: {e}")
if not torch.cuda.is_available(): sys.exit("FATAL: no CUDA. Use a GPU PyTorch image.")
print(f"OK torch {torch.__version__} CUDA {torch.version.cuda} "
      f"GPUs={torch.cuda.device_count()} {torch.cuda.get_device_name(0)}")
PY

echo "== 5. checkpoint =="
CKPT_DIR=""
case "$CKPT_ID" in
  ""|*"<"*|*">"*)
    echo "[bootstrap] CKPT_ID unset or still a placeholder."
    echo "            -> export CKPT_ID=<real-hf-id-no-angle-brackets> and re-run,"
    echo "               or set CHECKPOINT=/path manually before run_parallel.sh." ;;
  *)
    python3 -m pip install -q --root-user-action=ignore "huggingface_hub[cli]" || true
    CKPT_DIR="$WORK/ckpt"
    if [ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
      echo "[bootstrap] downloading $CKPT_ID"
      huggingface-cli download "$CKPT_ID" --local-dir "$CKPT_DIR" >/dev/null \
        || { echo "[bootstrap] WARN: checkpoint download failed; set CHECKPOINT manually."; CKPT_DIR=""; }
    else
      echo "[bootstrap] $CKPT_DIR non-empty, skipping download"
    fi ;;
esac

echo "== 6. data via PRIOR fetchers (REAL signatures; they write into arith-gating/data) =="
have() { [ -d "$1" ] && [ -n "$(ls -A "$1" 2>/dev/null)" ]; }
# CORD: the live HF mirror is Donut-style (ground_truth+image, no
# words/bboxes), so fetch_data.py REQUIRES --ocr (it self-reports this);
# tesseract was installed in step 1. "present" = annotations actually written.
if have "$AG/data/cord/test/annotations"; then echo "[bootstrap] cord present, skip"
  else python3 "$AG/scripts/fetch_data.py" --dataset cord --ocr || echo "[bootstrap] WARN: cord fetch failed (see error above)"; fi
if have "$AG/data/cord/dev/annotations"; then echo "[bootstrap] cord/dev present, skip"
  else python3 "$AG/scripts/fetch_cord_dev.py" || echo "[bootstrap] WARN: cord/dev fetch failed"; fi
if have "$AG/data/wild/test/annotations"; then echo "[bootstrap] wild present, skip"
  else python3 "$AG/scripts/fetch_wildreceipt.py" || echo "[bootstrap] WARN: wild fetch failed"; fi
echo "[bootstrap] NOTE: SROIE has no fetcher in these repos; supply it manually"
echo "            and add: export SROIE=sroie=/path/to/sroie/test"

echo "== 7. write .env.sh from DISCOVERED real paths =="
{
  [ -n "$CKPT_DIR" ] && echo "export CHECKPOINT=$CKPT_DIR"
  if have "$AG/data/cord/test/annotations"; then echo "export CORD=cord=$AG/data/cord/test"
  else echo "# WARNING: CORD test NOT produced (fetch failed); set CORD manually or experiments will fail" >&2; fi
  have "$AG/data/cord/dev/annotations"   && echo "export CORD_DEV=cord_dev=$AG/data/cord/dev"
  have "$AG/data/wild/test/annotations"  && echo "export WILDRECEIPT=wildreceipt=$AG/data/wild/test"
  echo "export HF_HOME=$HF_HOME"
} > "$PKG/.env.sh"
echo "[bootstrap] if CORD is absent below, the prior run's stale CORD env is invalid:"
echo "            run 'unset CORD' before 'source .env.sh'."
echo "---- .env.sh ----"; cat "$PKG/.env.sh"; echo "-----------------"

echo
echo "=================================================================="
echo "BOOTSTRAP DONE. Next:"
echo "  cd $PKG && source .env.sh"
echo "  [ -n \"\$CHECKPOINT\" ] || export CHECKPOINT=/path/to/ckpt   # if not auto-set"
echo "  bash run_parallel.sh"
echo "=================================================================="
