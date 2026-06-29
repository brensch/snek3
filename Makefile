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
# Pure-Rust /move API: one canonical live model under checkpoints/.
API_RUN_ID  ?= $(RUN_ID)
CHECKPOINT  ?= $(if $(API_RUN_ID),runs/$(API_RUN_ID)/state.pt,checkpoints/latest.pt)
MODEL       ?= checkpoints/latest.onnx
API_MAX_SIMS ?= 100000
API_TIMEOUT_MS ?= 500
API_DEADLINE_MARGIN_MS ?= 150
API_THREADS ?= 2
API_EVAL_CHUNK ?= 4096
API_MOVE_LOG_DIR ?= logs/api_moves
BATTLESNAKE ?= $(HOME)/go/bin/battlesnake
GAME_WIDTH ?= 11
GAME_HEIGHT ?= 11
GAME_TIMEOUT_MS ?= 500
GAME_DELAY_MS ?= 250
GAME_OUTPUT ?= local-vs-lan-game.json
GAME_LOCAL_NAME ?= local-8000
GAME_LOCAL_URL ?= http://127.0.0.1:8000
GAME_LAN_NAME ?= lan-8080
GAME_LAN_URL ?= http://192.168.1.22:8080

# Training defaults (all overridable)
GENERATIONS ?= 100000
TOTAL_GENERATIONS ?= 2500
CHUNK_GENERATIONS ?= 4
ADAPTIVE_EVERY ?= $(CHUNK_GENERATIONS)
SAMPLES     ?= 30000
COUNT       ?= 1024
SIMS        ?= 32
# AlphaZero net + game (single grid net, N-player FFA). Locked per run.
BOARD          ?= 11
NUM_SNAKES     ?= 4     # 4-player FFA subsumes 2-player as snakes die
C_PUCT         ?= 1.5
TRUNK_CHANNELS ?= 96
TRUNK_BLOCKS   ?= 8
DASH_PORT      ?= 8050  # in-process live dashboard port
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
RECENCY     ?= 2.0
BUFFER_SIZE ?= 500000
FILTERS     ?= 64
BLOCKS      ?= 6
EVAL_EVERY  ?= 5
EVAL_GAMES  ?= 32
RELATIVE_EVERY ?= 20
MAX_TURNS   ?= 200
SAMPLE_GAMES ?= 16
SAMPLE_EVERY ?= 1
KEEP_GAMES   ?= 40
RECORD_GAMES ?= 4
RECORD_EVERY ?= 20
RUN_ID      ?=
FRESH       ?=
ARGS        ?=

# Albatross trainer (proxy + response + UCT pool) defaults. These mirror the
# live fast-iteration "resp0" meta; override on the CLI as needed.
# steps_per_gen ~= ALB_SAMPLES / (ALB_COUNT * NUM_SNAKES). A gen plays this many
# board-turns, so games only reach the endgame if it exceeds typical game length.
# At 256x8000 that was ~16 -> games capped at ~16 turns (avg finished len ~11).
# 64x8000 -> ~62+ steps so games run to a natural finish (smaller batches, ~same
# compute budget). Want even longer/fuller endgames: lower ALB_COUNT or raise
# ALB_SAMPLES further.
ALB_COUNT      ?= 64    # parallel games per move (lower = longer games per gen)
ALB_SAMPLES    ?= 8000  # samples collected per generation
NUM_SNAKES     ?= 2
TAU_MIN        ?= 0.5   # proxy: low end of the per-episode temperature range
TAU_MAX        ?= 10.0  # proxy: high end (~optimal play)
RESPONSE_TAU   ?= 12.0  # response: the rational agent's fixed temperature tau_R
RESPONSE_AFTER ?= 0     # train the response net from gen 0 (vs warming the proxy first)
EVAL_OPP_TAU   ?= 1.0   # assumed opponent temperature for response eval
UCT_ITERS      ?= 200   # UCB sims for the CPU UCT pool opponent
ALB_EVAL_EVERY ?= 5     # full win-rate eval every N gens (the slow part; recording stays per-gen)
ALB_EVAL_GAMES ?= 16    # games per matchup in eval (batched; higher = less noise, slower)
ALB_TRAIN_STEPS ?= 128  # SGD steps/gen (lower = less reuse/overfit of stale buffer, faster gens)
ALB_RECENCY    ?= 2.0   # bias buffer sampling toward recent gens (1=uniform, >1=more recent)
LR             ?= 1e-3
# Egocentric obs are 21x21 (3.6x the cells of 11x11) so conv activations are big;
# keep Albatross GPU batches modest to stay within dedicated VRAM (watch
# gpu_peak_gb in the metrics and raise if you have headroom).
ALB_EVAL_BATCH ?= 2048
ALB_BATCH      ?= 1024
ALB_RECORD_GAMES ?= 4   # replays per (agent,opponent) matchup, recorded for the dashboard
ALB_RECORD_EVERY ?= 1   # record replays every N generations
ALB_MAX_TURNS  ?= 0     # 0 = games play until a snake dies (no artificial cap)
ALB_DRAW_VALUE ?= -0.9  # equilibrium-search terminal value of a draw (negative kills suicide-draws)
ALB_GENERATIONS ?= 0    # 0 = run forever until stopped via the dashboard/control API

.DEFAULT_GOAL := help
.PHONY: help venv build test test-rust test-py bench lint fmt train ui dashboard serve export-model api-build api api-run rungame api-docker clean clean-all

help: ## Show this help
	@echo "snek3 targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
	@echo
	@echo "Vars: GENERATIONS SAMPLES COUNT SIMS C_PUCT EXPLORATION_PROB DRAW_VALUE BOOTSTRAP_VALUE SKIP_SHORT_DRAW_TURNS EVAL_BATCH_SIZE SEARCH_THREADS TRAIN_STEPS BATCH_SIZE BUFFER_SIZE BOARD NUM_SNAKES TRUNK_CHANNELS TRUNK_BLOCKS MAX_TURNS SAMPLE_GAMES SAMPLE_EVERY KEEP_GAMES RUN_ID FRESH ARGS LR PORT SERVE_PORT CKPT TORCH_INDEX BATTLESNAKE GAME_LOCAL_URL GAME_LAN_URL GAME_OUTPUT"

venv: ## Create .venv and install all dependencies (incl. PyTorch)
	test -d $(VENV) || python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt
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
		--generations $(GENERATIONS) --board $(BOARD) --num-snakes $(NUM_SNAKES) \
		--samples $(SAMPLES) --count $(COUNT) --sims $(SIMS) --c-puct $(C_PUCT) \
		--trunk-channels $(TRUNK_CHANNELS) --trunk-blocks $(TRUNK_BLOCKS) \
		--exploration-prob $(EXPLORATION_PROB) \
		--draw-value $(DRAW_VALUE) --skip-short-draw-turns $(SKIP_SHORT_DRAW_TURNS) \
		$(if $(BOOTSTRAP_VALUE),--bootstrap-value,) \
		--eval-batch-size $(EVAL_BATCH_SIZE) \
		--search-threads $(SEARCH_THREADS) \
		--train-steps $(TRAIN_STEPS) --batch-size $(BATCH_SIZE) --recency $(RECENCY) \
		--buffer-size $(BUFFER_SIZE) \
		--max-turns $(MAX_TURNS) \
		--sample-games $(SAMPLE_GAMES) --sample-every $(SAMPLE_EVERY) --keep-games $(KEEP_GAMES) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(FRESH),--fresh,) $(ARGS)

server: build ## Start trainer + in-process dashboard on DASH_PORT. Resumes latest run unless RUN_ID is set.
	$(PY) -m azsnek.train \
		--serve --serve-port $(DASH_PORT) --runs-dir $(RUNS_DIR) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(FRESH),--fresh,) $(ARGS)

ui: ## Build the React dashboard UI (-> python/dashboard/static)
	cd python/dashboard/ui && npm install && npm run build

dashboard: ## Serve the live training dashboard on PORT (default 8050)
	SNEK_RUNS_DIR=$(RUNS_DIR) $(UVICORN) dashboard.app:app --host 127.0.0.1 --port $(PORT)

serve: build ## Run the Battlesnake server (set CKPT=path and matching FILTERS/BLOCKS)
	$(if $(CKPT),SNEK_CKPT=$(CKPT) ,)SNEK_FILTERS=$(FILTERS) SNEK_BLOCKS=$(BLOCKS) \
		$(UVICORN) server.main:app --host 0.0.0.0 --port $(SERVE_PORT)

export-model: build ## Export the AlphaZero net CHECKPOINT -> MODEL (.onnx) for the Rust API
	mkdir -p checkpoints
	$(if $(API_RUN_ID),cp $(CHECKPOINT) checkpoints/latest.pt.tmp && mv checkpoints/latest.pt.tmp checkpoints/latest.pt,)
	PYTHONPATH=python $(PY) scripts/export_model.py $(CHECKPOINT) $(MODEL)

viewer-build: ## Build the embedded game viewer frontend (crates/snek-server/viewer/dist)
	cd crates/snek-server/viewer && npm install && npm run build

api-build: viewer-build ## Compile the pure-Rust /move API server (release), embedding the viewer
	cargo build --release --manifest-path crates/snek-server/Cargo.toml

api: api-build ## Run the Rust /move API locally. Run `make export-model` first. Override API_RUN_ID, SERVE_PORT.
	ORT_DYLIB_PATH="$(shell ls $(VENV)/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so* 2>/dev/null | head -1)" \
	SNEK_MODEL=$(MODEL) SNEK_PORT=$(SERVE_PORT) SNEK_MAX_SIMS=$(API_MAX_SIMS) \
	SNEK_TIMEOUT_MS=$(API_TIMEOUT_MS) SNEK_DEADLINE_MARGIN_MS=$(API_DEADLINE_MARGIN_MS) \
	SNEK_THREADS=$(API_THREADS) SNEK_EVAL_CHUNK=$(API_EVAL_CHUNK) SNEK_MOVE_LOG_DIR=$(API_MOVE_LOG_DIR) \
	./crates/snek-server/target/release/snek-server

api-run: export-model api ## Export the selected run checkpoint, then start the Rust /move API.

rungame: ## Run a Battlesnake game between local API and LAN API, opening the viewer
	$(BATTLESNAKE) play \
		-W $(GAME_WIDTH) -H $(GAME_HEIGHT) \
		--name $(GAME_LOCAL_NAME) --url $(GAME_LOCAL_URL) \
		--name $(GAME_LAN_NAME) --url $(GAME_LAN_URL) \
		--timeout $(GAME_TIMEOUT_MS) \
		--browser \
		--delay $(GAME_DELAY_MS) \
		--output $(GAME_OUTPUT)

api-docker: ## Build the CPU-only Docker image for the Rust API (expects MODEL in repo root)
	docker build -f deploy/server.Dockerfile -t snek-api .

clean: ## Remove build artifacts and caches (keeps .venv and runs/)
	cargo clean
	rm -rf crates/snek-py/target .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

clean-all: clean ## Also remove .venv, runs/, and checkpoints/
	rm -rf $(VENV) runs checkpoints
