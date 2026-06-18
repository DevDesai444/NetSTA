# NetSTA developer Makefile. `make help` lists targets.

PY ?= python3
PIP ?= $(PY) -m pip
EPOCHS ?= 200

.PHONY: help install install-dev data train-gpu api web test lint clean

help:
	@echo "NetSTA targets:"
	@echo "  make install      — runtime dependencies"
	@echo "  make install-dev  — runtime + retrieval/agents/api/dev extras"
	@echo "  make data         — fetch benchmarks + build the real-netlist dataset"
	@echo "  make train-gpu    — train on a cloud GPU via Modal (EPOCHS=$(EPOCHS))"
	@echo "  make api          — run the FastAPI backend on :8000"
	@echo "  make web          — run the React dev server (proxies to the API)"
	@echo "  make test         — run the pytest suite"
	@echo "  make lint         — byte-compile all source modules"
	@echo "  make clean        — remove caches and local vector stores"

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[retrieval,agents,api,dev,ml-extras]"

data:
	bash scripts/fetch_benchmarks.sh
	$(PY) scripts/build_real_dataset.py --bench-root benchmarks --out data_real/graphs.pt

train-gpu:
	$(PY) -m modal run scripts/modal_train.py --split-mode random --epochs $(EPOCHS)

api:
	$(PY) -m uvicorn netsta.api:app --reload --port 8000

web:
	cd web && npm install && npm run dev

test:
	$(PY) -m pytest tests/ -q

lint:
	$(PY) -m py_compile $(shell find netsta scripts -name '*.py')
	@echo "lint: all source modules compile OK"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage coverage.xml circuit_embeddb netsta_vectordb netsta_faiss
	@echo "clean: removed caches"
