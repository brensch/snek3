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
GENERATIONS ?= 100000
TOTAL_GENERATIONS ?= 2500
CHUNK_GENERATIONS ?= 4
ADAPTIVE_EVERY ?= $(CHUNK_GENERATIONS)
SAMPLES     ?= 30000
COUNT       ?= 1024
SIMS        ?= 32
EXPLORATION_PROB ?= 0.15
DRAW_VALUE ?= -0.25
BOOTSTRAP_VALUE ?=
SKIP_SHORT_DRAW_TURNS ?= 0
DEPTH       ?= 2
TAU         ?= 30
ITERS       ?= 120
EVAL_BATCH_SIZE ?= 8192
SEARCH_THREADS ?= $(shell nproc 2>/dev/null || python3 -c 'import os; print(os.cpu_count() or 1)')
TRAIN_STEPS ?= 256
ADAPTIVE_TRAIN_STEPS ?= 256
BATCH_SIZE  ?= 2048
BUFFER_SIZE ?= 500000
FILTERS     ?= 64
BLOCKS      ?= 6
EVAL_EVERY  ?= 5
EVAL_GAMES  ?= 32
RELATIVE_EVERY ?= 20
MAX_TURNS   ?= 200
SAMPLE_GAMES ?= 16
SAMPLE_EVERY ?= 1
RECORD_GAMES ?= 4
RECORD_EVERY ?= 20
RUN_ID      ?=
FRESH       ?=
ARGS        ?=

# Albatross trainer (proxy + response + UCT pool) defaults.
ALB_COUNT      ?= 512   # equilibrium search is heavier per game than MCTS
NUM_SNAKES     ?= 2
TAU_MIN        ?= 0.5   # proxy: low end of the per-episode temperature range
TAU_MAX        ?= 10.0  # proxy: high end (~optimal play)
RESPONSE_TAU   ?= 12.0  # response: the rational agent's fixed temperature tau_R
RESPONSE_AFTER ?= 30    # start the response net after this many proxy generations
EVAL_OPP_TAU   ?= 1.0   # assumed opponent temperature for response eval
UCT_ITERS      ?= 200   # UCB sims for the CPU UCT pool opponent
LR             ?= 1e-3
# Egocentric obs are 21x21 (3.6x the cells of 11x11) so conv activations are big;
# keep Albatross GPU batches modest to stay within dedicated VRAM (watch
# gpu_peak_gb in the metrics and raise if you have headroom).
ALB_EVAL_BATCH ?= 2048
ALB_BATCH      ?= 1024

.DEFAULT_GOAL := help
.PHONY: help venv build test test-rust test-py bench lint fmt train albatross overnight adaptive ui dashboard serve audit clean clean-all

help: ## Show this help
	@echo "snek3 targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Vars: GENERATIONS TOTAL_GENERATIONS ADAPTIVE_EVERY SAMPLES COUNT SIMS EXPLORATION_PROB DRAW_VALUE BOOTSTRAP_VALUE SKIP_SHORT_DRAW_TURNS DEPTH TAU ITERS EVAL_BATCH_SIZE SEARCH_THREADS TRAIN_STEPS ADAPTIVE_TRAIN_STEPS BATCH_SIZE BUFFER_SIZE FILTERS BLOCKS EVAL_EVERY EVAL_GAMES RELATIVE_EVERY MAX_TURNS SAMPLE_GAMES SAMPLE_EVERY RECORD_GAMES RECORD_EVERY RUN_ID ARGS ALB_COUNT NUM_SNAKES TAU_MIN TAU_MAX RESPONSE_TAU RESPONSE_AFTER EVAL_OPP_TAU UCT_ITERS LR PORT SERVE_PORT CKPT TORCH_INDEX"

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
		--sims $(SIMS) \
		--exploration-prob $(EXPLORATION_PROB) \
		--draw-value $(DRAW_VALUE) --skip-short-draw-turns $(SKIP_SHORT_DRAW_TURNS) \
		$(if $(BOOTSTRAP_VALUE),--bootstrap-value,) \
		--depth $(DEPTH) --tau $(TAU) --iters $(ITERS) \
		--eval-batch-size $(EVAL_BATCH_SIZE) \
		--search-threads $(SEARCH_THREADS) \
		--train-steps $(TRAIN_STEPS) --batch-size $(BATCH_SIZE) \
		--buffer-size $(BUFFER_SIZE) \
		--filters $(FILTERS) --blocks $(BLOCKS) \
		--eval-every $(EVAL_EVERY) --eval-games $(EVAL_GAMES) \
		--relative-every $(RELATIVE_EVERY) \
		--max-turns $(MAX_TURNS) \
		--sample-games $(SAMPLE_GAMES) --sample-every $(SAMPLE_EVERY) \
		--record-games $(RECORD_GAMES) --record-every $(RECORD_EVERY) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(FRESH),--fresh,) $(ARGS)

albatross: build ## Full Albatross: temperature-conditioned proxy + best-response net + UCT opponent pool. Override TAU_MIN/TAU_MAX, RESPONSE_AFTER, RUN_ID, FRESH=1...
	$(PY) -m azsnek.train_albatross \
		--generations $(GENERATIONS) --num-snakes $(NUM_SNAKES) \
		--samples $(SAMPLES) --count $(ALB_COUNT) \
		--depth $(DEPTH) --iters $(ITERS) \
		--tau-min $(TAU_MIN) --tau-max $(TAU_MAX) \
		--response-tau $(RESPONSE_TAU) --response-after $(RESPONSE_AFTER) \
		--eval-opp-tau $(EVAL_OPP_TAU) --uct-iters $(UCT_ITERS) \
		--exploration-prob $(EXPLORATION_PROB) --max-turns $(MAX_TURNS) \
		--eval-batch-size $(ALB_EVAL_BATCH) \
		--filters $(FILTERS) --blocks $(BLOCKS) --lr $(LR) \
		--train-steps $(TRAIN_STEPS) --batch-size $(ALB_BATCH) --buffer-size $(BUFFER_SIZE) \
		--eval-every $(EVAL_EVERY) --eval-games $(EVAL_GAMES) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(FRESH),--fresh,) $(ARGS)

overnight: build ## Start a background overnight training run. Override TAU, GENERATIONS, SAMPLES, RUN_ID...
	TAU=$(TAU) GENERATIONS=$(GENERATIONS) SAMPLES=$(SAMPLES) COUNT=$(COUNT) \
	DEPTH=$(DEPTH) ITERS=$(ITERS) EVAL_BATCH_SIZE=$(EVAL_BATCH_SIZE) \
	SEARCH_THREADS=$(SEARCH_THREADS) TRAIN_STEPS=$(TRAIN_STEPS) BATCH_SIZE=$(BATCH_SIZE) \
	BUFFER_SIZE=$(BUFFER_SIZE) \
	FILTERS=$(FILTERS) BLOCKS=$(BLOCKS) \
	EVAL_EVERY=$(EVAL_EVERY) EVAL_GAMES=$(EVAL_GAMES) MAX_TURNS=$(MAX_TURNS) \
	EXPLORATION_PROB=$(EXPLORATION_PROB) DRAW_VALUE=$(DRAW_VALUE) BOOTSTRAP_VALUE=$(BOOTSTRAP_VALUE) SKIP_SHORT_DRAW_TURNS=$(SKIP_SHORT_DRAW_TURNS) \
	SAMPLE_GAMES=$(SAMPLE_GAMES) SAMPLE_EVERY=$(SAMPLE_EVERY) \
	RECORD_GAMES=$(RECORD_GAMES) RECORD_EVERY=$(RECORD_EVERY) \
	RUN_ID="$(RUN_ID)" FRESH="$(FRESH)" bash scripts/overnight_train.sh

adaptive: build ## Run adaptive training in the foreground; Ctrl-C stops it
	TOTAL_GENERATIONS=$(TOTAL_GENERATIONS) ADAPTIVE_EVERY=$(ADAPTIVE_EVERY) \
	TAU=$(TAU) SAMPLES=$(SAMPLES) COUNT=$(COUNT) DEPTH=$(DEPTH) ITERS=$(ITERS) \
	EVAL_BATCH_SIZE=$(EVAL_BATCH_SIZE) SEARCH_THREADS=$(SEARCH_THREADS) \
	TRAIN_STEPS=$(ADAPTIVE_TRAIN_STEPS) BATCH_SIZE=$(BATCH_SIZE) BUFFER_SIZE=$(BUFFER_SIZE) \
	FILTERS=$(FILTERS) BLOCKS=$(BLOCKS) EVAL_GAMES=64 MAX_TURNS=$(MAX_TURNS) \
	EXPLORATION_PROB=$(EXPLORATION_PROB) DRAW_VALUE=$(DRAW_VALUE) BOOTSTRAP_VALUE=$(BOOTSTRAP_VALUE) SKIP_SHORT_DRAW_TURNS=$(SKIP_SHORT_DRAW_TURNS) \
	SAMPLE_GAMES=$(SAMPLE_GAMES) SAMPLE_EVERY=$(SAMPLE_EVERY) \
	RECORD_GAMES=$(RECORD_GAMES) RECORD_EVERY=$(RECORD_EVERY) \
	RUN_ID="$(RUN_ID)" FRESH="$(FRESH)" ARGS="$(ARGS)" bash scripts/adaptive_train.sh

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
