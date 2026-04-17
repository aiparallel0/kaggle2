#!/usr/bin/env bash
# kaggle2/scripts/vastai_bootstrap.sh
# One-shot bootstrap for a fresh vast.ai PyTorch instance. Idempotent.
#
# Usage on a vast.ai PyTorch image (Ubuntu 22.04+, CUDA 12+):
#   cd /workspace
#   git clone https://github.com/aiparallel0/kaggle2.git && cd kaggle2
#   bash scripts/vastai_bootstrap.sh
#   make all   # trains DONUT + YOLO+TrOCR+Attention, evaluates, writes paper
#
# This script intentionally does NOT talk to the vast.ai API, provision
# machines, or run over SSH. Spin the instance up via the vast.ai web UI
# (PyTorch template, >=24 GB GPU, >=50 GB disk), open its terminal / Jupyter,
# and paste the three commands above.
set -euo pipefail

log() { printf "\033[1;36m[bootstrap]\033[0m %s\n" "$*"; }

log "Python version: $(python --version 2>&1)"
log "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "ERROR: no GPU detected. kaggle2 training needs CUDA." >&2
    exit 1
}

log "Installing system deps (texlive for the paper stage)"
# texlive is only needed for the final PDF compile; main.py::_compile_paper_pdf
# already warns and emits the .tex when pdflatex is missing. So an apt-get
# failure (Ubuntu mirror timeout, sandboxed network, offline host, ...) must
# NOT abort the bootstrap — otherwise training/eval never run. Retry a few
# times, then degrade to a warning.
install_texlive() {
    local attempt
    for attempt in 1 2 3; do
        if apt-get update -qq \
            && apt-get install -y --no-install-recommends \
                texlive-latex-base texlive-latex-recommended texlive-latex-extra \
                texlive-fonts-recommended texlive-bibtex-extra biber git >/dev/null; then
            return 0
        fi
        log "apt-get attempt ${attempt}/3 failed, retrying in 5s..."
        sleep 5
    done
    return 1
}

if ! command -v pdflatex >/dev/null; then
    if install_texlive; then
        log "texlive installed."
    else
        log "WARNING: could not install texlive (apt-get failed, likely a"
        log "         transient mirror / network issue). Continuing without"
        log "         it — 'make paper' will still write report/paper_filled.tex"
        log "         but will skip PDF compilation. Re-run this script once"
        log "         connectivity to archive.ubuntu.com is restored to get"
        log "         the final PDF."
    fi
fi

log "Installing Python requirements"
pip install --upgrade pip
pip install -r requirements.txt

log "Running static checks (mypy + ruff) — the kaggle2 'test suite'"
make check

log "Ready. Launch training with: make all"
log "Intermediate outputs land in ./results/, final paper in report/paper_filled.pdf"
