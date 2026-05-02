.PHONY: all train eval paper check test clean serve \
        train-pipeline eval-pipeline pipeline \
        pack pack-full unpack runs-list latest clean-runs \
        check_artefacts

.DELETE_ON_ERROR:

all: check train eval paper check_artefacts

check:
	# Bifurcation cutover: --ignore-missing-imports tolerates the residual
	# broken imports between top-level shared substrate and the moved
	# Paper 2 / Paper 3 modules.  Tracked as deferred work in README.md
	# status section + AGENTS.md.
	mypy --strict --ignore-missing-imports core/ data/ models/ report/ stages/ main.py
	ruff check .
	python -c "from core.types import Receipt, Metrics; from core.config import load_config"

check-paper2:
	# Paper 2 self-contained mypy + import smoke (deferred until imports fixed).
	mypy --strict --ignore-missing-imports paper2/
	ruff check paper2/

check-paper3:
	# Paper 3 self-contained mypy + import smoke (deferred until imports fixed).
	mypy --strict --ignore-missing-imports paper3/
	ruff check paper3/

# v4 build gate: scans the rendered paper.tex for unresolved \VAR{},
# dangling Sec.~?? / Fig.~?? refs, undefined-citation [?] markers,
# unresolved \MissingCell{} in strict mode, and zero-byte PDFs in
# the figures dir.  Non-zero exit fails ``make all`` so a half-empty
# PDF can never silently ship.  Uses the latest runs/<run_id>/ by
# default; override via ``make check_artefacts PAPER=path FIGS=dir``.
check_artefacts:
	python -m report.check_artefacts $(if $(PAPER),--paper $(PAPER)) $(if $(FIGS),--figures $(FIGS))

test:
	python -m pytest tests/ -v

train:
	python main.py --stage train

eval:
	python main.py --stage eval

paper:
	python main.py --stage paper

# Pipeline-only targets — skip DONUT training/eval entirely (Phase 1 / GPU-constrained runs).
# Equivalent to setting skip_donut=true in config.json, but without editing the file.
train-pipeline:
	python main.py --stage train --skip-donut

eval-pipeline:
	python main.py --stage eval --skip-donut

pipeline: train-pipeline eval-pipeline paper

clean:
	rm -rf data/sroie_cache/ $(shell find . -name '__pycache__' -type d)

# Run artefact housekeeping (see core/runlayout.py for the layout contract).
# ``runs-list`` enumerates every run on disk; ``latest`` prints the newest;
# ``pack`` tars it into a single .tar.zst + sha256 for vast.ai → Copilot
# round-trips; ``unpack`` is the inverse (verifies per-file sha256 against
# MANIFEST.json); ``clean-runs`` deletes every run directory.
runs-list:
	@ls -1t runs 2>/dev/null || echo "(no runs/ directory yet)"

latest:
	@python -c "from core.runlayout import latest_run; p=latest_run('runs'); print(p or '(none)')"

pack:
	bash scripts/pack_run.sh --light

# ``pack-full`` includes the heavy model checkpoints (DONUT ~770 MiB,
# TrOCR ~300 MiB, YOLO weights + data mirror).  Use only when the
# checkpoints themselves need to ship; otherwise prefer ``make pack``
# (the default, which drops files > 1 MiB and writes EXCLUDED.txt).
pack-full:
	bash scripts/pack_run.sh --full

unpack:
	@test -n "$(ARCHIVE)" || { echo "usage: make unpack ARCHIVE=<path>.tar.zst"; exit 1; }
	bash scripts/unpack_run.sh "$(ARCHIVE)"

clean-runs:
	rm -rf runs/

# Run the demo website — drop a receipt, see DONUT extract the fields.
# Uses the fine-tuned checkpoint in results/donut if present, otherwise
# falls back to config.base_model so the page works before training.
serve:
	python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --root-path /teb2
