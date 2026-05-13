# NetSTA developer Makefile. `make help` lists targets.

PY ?= python3
PIP ?= $(PY) -m pip

# Default training knobs — keep CPU-friendly so `make train` finishes in
# minutes, not hours. Override on the command line: `make train EPOCHS=200`.
NUM_CIRCUITS ?= 200
EPOCHS       ?= 10
TASKS        ?= slack,critical_path,congestion,drc

.PHONY: help install install-dev train test test-fast demo lint clean

help:
	@echo "NetSTA targets:"
	@echo "  make install       — install runtime dependencies"
	@echo "  make install-dev   — install runtime + test/dev dependencies"
	@echo "  make train         — quick training run (NUM_CIRCUITS=$(NUM_CIRCUITS), EPOCHS=$(EPOCHS))"
	@echo "  make test          — run the full pytest suite with verbose output"
	@echo "  make test-fast     — skip slow tests (RAG / similarity) — pure model+circuit"
	@echo "  make demo          — launch the Streamlit demo (http://localhost:8501)"
	@echo "  make lint          — byte-compile all source modules"
	@echo "  make clean         — remove __pycache__, .pytest_cache, runtime caches"

install:
	$(PIP) install -r requirements.txt

install-dev: install
	$(PIP) install -e ".[rag,demo,dev,ml-extras]"

train:
	$(PY) -m netsta.train \
	    --tasks $(TASKS) \
	    --num-circuits $(NUM_CIRCUITS) \
	    --epochs $(EPOCHS)

test:
	$(PY) -m pytest tests/ -v --tb=short

test-fast:
	$(PY) -m pytest tests/test_model.py tests/test_circuit_gen.py \
	                tests/test_sta.py tests/test_congestion.py -v --tb=short

demo:
	$(PY) -m streamlit run app.py

lint:
	$(PY) -m py_compile $(shell find netsta netsta scripts -name '*.py')
	@echo "lint: all source modules compile OK"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	rm -rf .pytest_cache .coverage coverage.xml
	rm -rf circuit_embeddb netsta_vectordb
	@echo "clean: removed caches"
