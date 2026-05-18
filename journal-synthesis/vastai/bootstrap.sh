#!/usr/bin/env bash
# =====================================================================
# FRESH vast.ai INSTANCE BOOTSTRAP  (copy-paste, idempotent)
# =====================================================================
# One-time setup on a fresh vast.ai PyTorch instance for the journal
# experiment package. Safe to re-run (clones -> pull, fetch -> skip if
# data already present).
#
# HONESTY / SCOPE:
#  - Does NOT pip-install torch/torchvision: vast.ai PyTorch images ship
#    a matched pair; pip torch breaks torchvision's C-extension. We use
#    the image's torch (verified below).
#  - The KIE checkpoint id is NOT baked in. You pass CKPT_ID (a public
#    HF Donut CORD-v2 checkpoint of your choosing); nothing is fabricated
#    and no model identifier is stored in the repo.
#  - Fetches data ONLY via the prior repos' existing fetch scripts; this
#    package does not re-implement data fetching.
#
# USAGE (edit the three URLs / CKPT_ID, then paste the whole block):
#   export REPO_URL=https://github.com/aiparallel0/kaggle2.git
#   export ARITH_URL=https://github.com/aiparallel0/arith-gating.git
#   export TRIOLOGY_URL=https://github.com/aiparallel0/triology.git
#   export CKPT_ID=<hf-donut-cord-v2-checkpoint-id>      # you choose
#   bash bootstrap.sh
# =====================================================================
set -euo pipefail

BRANCH="${BRANCH:-claude/prepare-papers-repos-4LUdJ}"
WORK="${WORK:-/workspace}"
DATA="${DATA:-/data}"
REPO_URL="${REPO_URL:?set REPO_URL to the kaggle2 git remote (https)}"
ARITH_URL="${ARITH_URL:?set ARITH_URL to the arith-gating git remote}"
TRIOLOGY_URL="${TRIOLOGY_URL:?set TRIOLOGY_URL to the triology git remote}"
CKPT_ID="${CKPT_ID:-}"

export HF_HOME="${HF_HOME:-$DATA/hf}"          # big-disk HF cache
export PYTHONUNBUFFERED=1
mkdir -p "$WORK" "$DATA" "$HF_HOME"

clone_or_pull() {  # url dir
  local url="$1" dir="$2"
  if [ -d "$dir/.git" ]; then
    echo "[bootstrap] updating $dir"
    git -C "$dir" fetch -q origin "$BRANCH" && git -C "$dir" checkout -q "$BRANCH" \
      && git -C "$dir" pull -q --rebase origin "$BRANCH" || true
  else
    echo "[bootstrap] cloning $url -> $dir"
    git clone -q --branch "$BRANCH" --single-branch "$url" "$dir" \
      || git clone -q "$url" "$dir"
  fi
}

echo "== 1. system deps =="
( sudo apt-get update -qq && sudo apt-get install -y -qq git python3-pip tesseract-ocr ) \
  || echo "[bootstrap] apt step skipped/non-fatal (likely already present)"

echo "== 2. repos =="
clone_or_pull "$REPO_URL"     "$WORK/kaggle2"
clone_or_pull "$ARITH_URL"    "$WORK/arith-gating"
clone_or_pull "$TRIOLOGY_URL" "$WORK/triology"
PKG="$WORK/kaggle2/journal-synthesis/vastai"

echo "== 3. python deps (NOT torch) =="
python3 -m pip install -q --upgrade pip
python3 -m pip install -q -r "$PKG/requirements.txt"

echo "== 4. GPU sanity (fail loud if no CUDA) =="
python3 - <<'PY'
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"FATAL: torch not importable from the image: {e}")
if not torch.cuda.is_available():
    sys.exit("FATAL: CUDA not available. Pick a GPU instance / PyTorch image.")
print(f"OK torch {torch.__version__}  CUDA {torch.version.cuda}  "
      f"GPUs={torch.cuda.device_count()}  "
      f"{torch.cuda.get_device_name(0)}")
PY

echo "== 5. checkpoint =="
if [ -n "$CKPT_ID" ]; then
  python3 -m pip install -q "huggingface_hub[cli]" >/dev/null 2>&1 || true
  CKPT_DIR="$DATA/ckpt"
  if [ ! -d "$CKPT_DIR" ] || [ -z "$(ls -A "$CKPT_DIR" 2>/dev/null)" ]; then
    echo "[bootstrap] downloading checkpoint $CKPT_ID"
    huggingface-cli download "$CKPT_ID" --local-dir "$CKPT_DIR" >/dev/null
  else
    echo "[bootstrap] checkpoint dir non-empty, skipping download"
  fi
  echo "export CHECKPOINT=$CKPT_DIR" > "$PKG/.env.sh"
else
  echo "[bootstrap] CKPT_ID not set: set CHECKPOINT manually before running."
  : > "$PKG/.env.sh"
fi

echo "== 6. data via PRIOR repos' fetchers (skip if present) =="
fetch() {  # marker_dir  command...
  local marker="$1"; shift
  if [ -d "$marker" ] && [ -n "$(ls -A "$marker" 2>/dev/null)" ]; then
    echo "[bootstrap] $marker already populated, skipping"
  else
    echo "[bootstrap] fetching -> $marker"
    "$@"
  fi
}
AG="$WORK/arith-gating/scripts"
fetch "$DATA/cord"        python3 "$AG/fetch_data.py"      --dataset cord --out "$DATA/cord"     || true
fetch "$DATA/cord_dev"    python3 "$AG/fetch_cord_dev.py"  --out "$DATA/cord_dev"                || true
fetch "$DATA/wildreceipt" python3 "$AG/fetch_wildreceipt.py" --out "$DATA/wildreceipt"          || true
echo "[bootstrap] (SROIE: fetch via the same arith-gating path / triology sroie helper if used)"

cat >> "$PKG/.env.sh" <<EOF
export CORD=cord=$DATA/cord/test
export WILDRECEIPT=wildreceipt=$DATA/wildreceipt/test
# export SROIE=sroie=$DATA/sroie/test   # uncomment once SROIE fetched
export HF_HOME=$HF_HOME
EOF

echo
echo "=================================================================="
echo "BOOTSTRAP DONE. Next:"
echo "  cd $PKG"
echo "  source .env.sh        # exports CHECKPOINT/CORD/WILDRECEIPT/..."
echo "  bash run_parallel.sh  # GPU-aware, resumable runner"
echo "(or: bash run_all.sh for the simple ordered driver)"
echo "=================================================================="
