.PHONY: setup weights lint test run clean

PY ?= python3
VENV ?= .venv

setup:
	$(PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip
	$(VENV)/bin/pip install -r requirements/dev.txt
	$(VENV)/bin/pre-commit install

weights:
	bash scripts/download_weights.sh

lint:
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

test:
	$(VENV)/bin/pytest

run:
	$(VENV)/bin/footy run --video $(VIDEO) --config configs/pipeline.yaml --match $(MATCH)

clean:
	rm -rf data/interim/* data/processed/* .pytest_cache .ruff_cache
