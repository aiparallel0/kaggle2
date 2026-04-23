.PHONY: all train eval paper check test clean serve \
        train-pipeline eval-pipeline pipeline

.DELETE_ON_ERROR:

all: check train eval paper

check:
	mypy --strict core/ data/ models/ report/ stages/ main.py
	ruff check .
	python -c "from core.types import Receipt, Metrics; from core.config import load_config"

test:
	python -m pytest tests/ -v

train:
	python main.py --stage train

eval:
	python main.py --stage eval

paper:
	python main.py --stage paper
	@test -s report/paper_filled.pdf || { \
		echo "ERROR: report/paper_filled.pdf missing or empty after 'make paper'."; \
		echo "       Install a LaTeX engine (tectonic or pdflatex) via"; \
		echo "       scripts/vastai_bootstrap.sh and rerun."; \
		exit 1; }

# Pipeline-only targets — skip DONUT training/eval entirely (Phase 1 / GPU-constrained runs).
# Equivalent to setting skip_donut=true in config.json, but without editing the file.
train-pipeline:
	python main.py --stage train --skip-donut

eval-pipeline:
	python main.py --stage eval --skip-donut

pipeline: train-pipeline eval-pipeline paper

clean:
	rm -rf results/ data/sroie_cache/ $(shell find . -name '__pycache__' -type d)

# Run the demo website — drop a receipt, see DONUT extract the fields.
# Uses the fine-tuned checkpoint in results/donut if present, otherwise
# falls back to config.base_model so the page works before training.
serve:
	python -m uvicorn app.server:app --host 0.0.0.0 --port 8000 --root-path /teb2
