# snek3 — common tasks. Run `make` (or `make help`) to list targets.
# Override variables on the CLI, e.g. `make train GENERATIONS=50 SAMPLES=20000`.

VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
MATURIN := $(VENV)/bin/maturin
UVICORN := $(VENV)/bin/uvicorn
PYTEST  := $(PY) -m pytest

# PyTorch wheel index (RTX 50-series/Blackwell needs cu128). Use
# https://download.pytorch.org/whl/cpu for a CPU-only install.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu128

# Dashboard / server
RUNS_DIR    ?= runs
PORT        ?= 8050
SERVE_PORT  ?= 8000
CKPT        ?=

# Training defaults (all overridable)
GENERATIONS ?= 30
SAMPLES     ?= 12000
COUNT       ?= 128
DEPTH       ?= 2
TAU         ?= 30
ITERS       ?= 120
SEARCH_THREADS ?= $(shell nproc 2>/dev/null || python3 -c 'import os; print(os.cpu_count() or 1)')
FILTERS     ?= 64
BLOCKS      ?= 6
EVAL_EVERY  ?= 5
EVAL_GAMES  ?= 200
RECORD_GAMES ?= 8
RECORD_EVERY ?= 1
RUN_ID      ?=
FRESH       ?=
ARGS        ?=

.DEFAULT_GOAL := help
.PHONY: help venv build test test-rust test-py bench lint fmt train overnight ui dashboard serve audit clean clean-all

help: ## Show this help
	@echo "snek3 targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Vars: GENERATIONS SAMPLES COUNT DEPTH TAU ITERS SEARCH_THREADS FILTERS BLOCKS EVAL_EVERY EVAL_GAMES RECORD_GAMES RECORD_EVERY RUN_ID ARGS PORT SERVE_PORT CKPT TORCH_INDEX"

venv: ## Create .venv and install all dependencies (incl. PyTorch)
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -U pip maturin numpy pytest fastapi uvicorn httpx pylint
	$(PIP) install torch --index-url $(TORCH_INDEX)
	@echo "venv ready. Next: make build"

build: ## Compile the Rust extension into the venv (maturin develop)
	$(MATURIN) develop --release

test: test-rust test-py ## Run all tests (Rust + Python)

test-rust: ## Run the Rust test suite
	cargo test

test-py: build ## Run the Python test suite
	$(PYTEST) python/tests -q

bench: ## Benchmark rules-engine step throughput
	cargo bench -p snek-core

lint: ## Lint Python (import-error, no-member) and Rust (clippy)
	$(PY) -m pylint --disable=all --enable=import-error,no-member \
		python/azsnek python/server python/dashboard python/tests
	cargo clippy --workspace --all-targets

fmt: ## Format Rust code
	cargo fmt

train: build ## Train (auto-resumes RUN_ID if it has saved state). Override GENERATIONS, SAMPLES, RUN_ID, FRESH=1, ARGS...
	$(PY) -m azsnek.train \
		--generations $(GENERATIONS) --samples $(SAMPLES) --count $(COUNT) \
		--depth $(DEPTH) --tau $(TAU) --iters $(ITERS) \
		--search-threads $(SEARCH_THREADS) \
		--filters $(FILTERS) --blocks $(BLOCKS) \
		--eval-every $(EVAL_EVERY) --eval-games $(EVAL_GAMES) \
		--record-games $(RECORD_GAMES) --record-every $(RECORD_EVERY) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(FRESH),--fresh,) $(ARGS)

overnight: build ## Start a background overnight training run. Override TAU, GENERATIONS, SAMPLES, RUN_ID...
	TAU=$(TAU) GENERATIONS=$(GENERATIONS) SAMPLES=$(SAMPLES) COUNT=$(COUNT) \
	DEPTH=$(DEPTH) ITERS=$(ITERS) SEARCH_THREADS=$(SEARCH_THREADS) \
	FILTERS=$(FILTERS) BLOCKS=$(BLOCKS) \
	EVAL_EVERY=$(EVAL_EVERY) EVAL_GAMES=$(EVAL_GAMES) \
	RECORD_GAMES=$(RECORD_GAMES) RECORD_EVERY=$(RECORD_EVERY) \
	RUN_ID="$(RUN_ID)" FRESH="$(FRESH)" bash scripts/overnight_train.sh

ui: ## Build the React dashboard UI (-> python/dashboard/static)
	cd python/dashboard/ui && npm install && npm run build

dashboard: ## Serve the live training dashboard on PORT (default 8050)
	SNEK_RUNS_DIR=$(RUNS_DIR) $(UVICORN) dashboard.app:app --host 127.0.0.1 --port $(PORT)

serve: build ## Run the Battlesnake server (set CKPT=path and matching FILTERS/BLOCKS)
	$(if $(CKPT),SNEK_CKPT=$(CKPT) ,)SNEK_FILTERS=$(FILTERS) SNEK_BLOCKS=$(BLOCKS) \
		$(UVICORN) server.main:app --host 0.0.0.0 --port $(SERVE_PORT)

audit: ## Run the full end-to-end audit script
	bash scripts/audit.sh

clean: ## Remove build artifacts and caches (keeps .venv and runs/)
	cargo clean
	rm -rf crates/snek-py/target .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-all: clean ## Also remove .venv, runs/, and checkpoints/
	rm -rf $(VENV) runs checkpoints
