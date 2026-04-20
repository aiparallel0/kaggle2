.PHONY: all train eval paper check clean

.DELETE_ON_ERROR:

all: check train eval paper

check:
	mypy --strict core/ data/ models/ report/ stages.py main.py
	ruff check .
	python -c "from core.types import Receipt, Metrics; from core.config import load_config"

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

clean:
	rm -rf results/ data/sroie_cache/ $(shell find . -name '__pycache__' -type d)
