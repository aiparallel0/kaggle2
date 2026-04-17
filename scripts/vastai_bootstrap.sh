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
if [ "${VERBOSE:-0}" = "1" ]; then set -x; fi

log() { printf "\033[1;36m[bootstrap]\033[0m %s\n" "$*"; }

log "Python version: $(python --version 2>&1)"
log "GPU(s):"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
    echo "ERROR: no GPU detected. kaggle2 training needs CUDA." >&2
    exit 1
}

log "Installing LaTeX toolchain for the paper stage"
# We use Tectonic rather than apt-installed texlive. Tectonic is a single
# self-contained Rust/musl binary that bundles a full TeX Live on demand and
# runs bibtex/rerun automatically. Installing it only requires a download
# from GitHub releases — the same network that already cloned this repo —
# so it works on hosts where the Ubuntu archive mirror is unreachable
# (e.g. transient vast.ai network issues with archive.ubuntu.com).
TECTONIC_VERSION="${TECTONIC_VERSION:-0.15.0}"
TECTONIC_BIN="/usr/local/bin/tectonic"

install_tectonic() {
    if command -v tectonic >/dev/null; then
        log "tectonic already installed: $(tectonic --version 2>&1 | head -n1)"
        return 0
    fi
    local arch
    arch="$(uname -m)"
    case "$arch" in
        x86_64|amd64) arch="x86_64" ;;
        aarch64|arm64) arch="aarch64" ;;
        *) log "unknown arch '$arch' for tectonic; skipping"; return 1 ;;
    esac
    local url="https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-${arch}-unknown-linux-musl.tar.gz"
    local tmp
    tmp="$(mktemp -d)"
    local attempt
    for attempt in 1 2 3; do
        if curl -fsSL --retry 3 --retry-delay 2 -o "${tmp}/tectonic.tar.gz" "$url"; then
            if tar -xzf "${tmp}/tectonic.tar.gz" -C "$tmp" tectonic; then
                install -m 0755 "${tmp}/tectonic" "$TECTONIC_BIN"
                rm -rf "$tmp"
                log "tectonic installed: $(tectonic --version 2>&1 | head -n1)"
                return 0
            fi
        fi
        log "tectonic download attempt ${attempt}/3 failed, retrying in 5s..."
        sleep 5
    done
    rm -rf "$tmp"
    return 1
}

# Best-effort apt texlive install — only if tectonic is somehow unavailable
# AND the apt mirror is reachable. Failures here are NOT fatal: main.py's
# paper stage works with either tectonic or pdflatex, and degrades to
# emitting the .tex source if neither is present.
install_apt_texlive() {
    local attempt
    for attempt in 1 2 3; do
        if apt-get update -qq \
            && apt-get install -y --no-install-recommends \
                texlive-latex-base texlive-latex-recommended texlive-latex-extra \
                texlive-fonts-recommended texlive-bibtex-extra biber >/dev/null; then
            return 0
        fi
        log "apt-get attempt ${attempt}/3 failed, retrying in 5s..."
        sleep 5
    done
    return 1
}

if command -v tectonic >/dev/null || command -v pdflatex >/dev/null; then
    log "LaTeX engine already present — skipping install."
elif install_tectonic; then
    :
elif install_apt_texlive; then
    log "texlive installed via apt as fallback."
else
    log "WARNING: could not install tectonic (GitHub download failed) and"
    log "         apt-get could not reach the Ubuntu archive. Continuing;"
    log "         'make paper' will still write report/paper_filled.tex but"
    log "         will skip PDF compilation until a LaTeX engine is available."
fi

log "Installing Python requirements"
pip install --upgrade pip
pip install -r requirements.txt

log "Running static checks (mypy + ruff) — the kaggle2 'test suite'"
make check

log "LaTeX engine summary:"
if command -v tectonic >/dev/null; then
    log "  tectonic: $(tectonic --version 2>&1 | head -n1)"
elif command -v pdflatex >/dev/null; then
    log "  pdflatex: $(pdflatex --version 2>&1 | head -n1)"
else
    log "  WARN: no LaTeX engine (tectonic or pdflatex) — 'make paper' will only produce .tex"
fi

log "Ready. Launch training with: make all"
log "Intermediate outputs land in ./results/, final paper in report/paper_filled.pdf"
