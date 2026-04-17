.PHONY: all train eval paper check test clean

all: check test train eval paper

check:
	mypy --strict core/ data/ models/ report/ main.py
	ruff check .
	python -c "from core.types import Receipt, Metrics; from core.config import load_config"

test:
	python -m pytest -q tests/

train:
	python main.py --stage train

eval:
	python main.py --stage eval

paper:
	python main.py --stage paper

clean:
	rm -rf results/ data/sroie_cache/ $(shell find . -name '__pycache__' -type d)
