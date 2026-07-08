# snek3 common tasks. Override variables on the CLI, e.g. `make train RUN_ID=dev`.

RUNS_DIR ?= runs
PORT ?= 8050
RUN_ID ?=
FRESH ?=
BIND ?= 127.0.0.1:$(PORT)

SERVE_PORT ?= 8000
MODEL ?= net.safetensors
API_MAX_SIMS ?= 100000
API_TIMEOUT_MS ?= 500
API_DEADLINE_MARGIN_MS ?= 150
API_THREADS ?= 2
API_EVAL_CHUNK ?= 4096
API_MOVE_LOG_DIR ?= logs/api_moves

PYTHON ?= .venv/bin/python
TORCH_SITE ?= $(shell $(PYTHON) -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)
TORCH_LIB ?= $(TORCH_SITE)/torch/lib
NVIDIA_LIBS ?= $(shell find $(TORCH_SITE)/nvidia -name lib -type d 2>/dev/null | tr '\n' ':')
LIBTORCH_PRELOAD ?= $(TORCH_LIB)/libtorch_global_deps.so:$(TORCH_LIB)/libtorch_cuda.so
LIBTORCH_ENV := PYTHON=$(PYTHON) LIBTORCH_USE_PYTORCH=1 LIBTORCH_BYPASS_VERSION_CHECK=1 LD_PRELOAD="$(LIBTORCH_PRELOAD)$${LD_PRELOAD:+:$$LD_PRELOAD}" LD_LIBRARY_PATH="$(TORCH_LIB):$(NVIDIA_LIBS)$$LD_LIBRARY_PATH"

.DEFAULT_GOAL := help
.PHONY: help test test-rust fmt lint train train-build tunnel frontend frontend-build api-build api baseline rungame rungamelocal clean

help: ## Show this help
	@echo "snek3 targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

test: test-rust ## Run the Rust test suite

test-rust: ## Run top-level Rust tests
	cargo test

fmt: ## Format Rust code
	cargo fmt --all
	cargo fmt --manifest-path crates/snek-train/Cargo.toml

lint: ## Lint Rust code
	cargo clippy --workspace --all-targets
	$(LIBTORCH_ENV) cargo clippy --manifest-path crates/snek-train/Cargo.toml --all-targets

train-build: ## Build the Rust trainer/API binary
	$(LIBTORCH_ENV) cargo build --release --manifest-path crates/snek-train/Cargo.toml

train: train-build ## Run the Rust trainer/API. Add START=1 to begin immediately.
	$(LIBTORCH_ENV) ./crates/snek-train/target/release/snek-train \
		--bind $(BIND) --runs-dir $(RUNS_DIR) \
		$(if $(RUN_ID),--run-id $(RUN_ID),) $(if $(FRESH),--fresh,) $(if $(START),--start,)

tunnel: ## Expose the dashboard over your tailnet (needs TS_AUTHKEY on first run)
	DASH_PORT=$(PORT) deploy/tailscale-dashboard.sh

frontend: ## Run the standalone Vite frontend (proxies /api to PORT)
	cd frontend && npm install && npm run dev

frontend-build: ## Build the standalone Vite frontend
	cd frontend && npm install && npm run build

api-build: ## Build the Battlesnake /move API server
	$(LIBTORCH_ENV) cargo build --release --manifest-path crates/snek-server/Cargo.toml

api: api-build ## Run the Battlesnake /move API server with an existing safetensors model
	$(LIBTORCH_ENV) \
	SNEK_MODEL=$(MODEL) SNEK_PORT=$(SERVE_PORT) SNEK_MAX_SIMS=$(API_MAX_SIMS) \
	SNEK_TIMEOUT_MS=$(API_TIMEOUT_MS) SNEK_DEADLINE_MARGIN_MS=$(API_DEADLINE_MARGIN_MS) \
	SNEK_THREADS=$(API_THREADS) SNEK_EVAL_CHUNK=$(API_EVAL_CHUNK) SNEK_MOVE_LOG_DIR=$(API_MOVE_LOG_DIR) \
	./crates/snek-server/target/release/snek-server

A_NAME ?= challenger
A_URL ?= http://192.168.1.22:8080
B_NAME ?= snek3-api
B_URL ?= http://192.168.1.8:8000
# Seats 3+4 default to the old Cloud Run snakes; set C_URL/D_URL empty for a
# 1v1, or use rungame4 to fill them with local voronoi baselines instead.
C_NAME ?= cloud-green
C_URL ?= https://snake-233u62v37a-uk.a.run.app
D_NAME ?= cloud-orange
D_URL ?= https://snake-uvyj55g6wa-wl.a.run.app
BASELINE_PORT ?= 8100
WSL_BROWSER ?= google-chrome
# comm name of the running browser ($(WSL_BROWSER) is a wrapper that execs this)
WSL_BROWSER_PROC ?= chrome

baseline: ## Serve the voronoi baseline as a Battlesnake (one port serves any number of seats)
	cargo build --release -p snek-heuristic --features server --bin snek-baseline
	SNEK_BASELINE_PORT=$(BASELINE_PORT) ./target/release/snek-baseline

rungame: ## Play A_URL vs B_URL (plus C_URL/D_URL if set) on the browser board
	@pgrep -x $(WSL_BROWSER_PROC) >/dev/null || { \
		echo "starting $(WSL_BROWSER) in WSL (board glitches if it cold-starts mid-game)"; \
		nohup $(WSL_BROWSER) about:blank >/dev/null 2>&1 & sleep 3; }
	battlesnake play -W 11 -H 11 \
		--name $(A_NAME) --url $(A_URL) \
		--name $(B_NAME) --url $(B_URL) \
		$(if $(C_URL),--name $(C_NAME) --url $(C_URL),) \
		$(if $(D_URL),--name $(D_NAME) --url $(D_URL),) \
		--browser

rungamelocal: ## rungame with no cloud seats: players 3+4 are local voronoi snakes (run `make baseline` first)
	$(MAKE) rungame C_NAME=voronoi1 C_URL=http://localhost:$(BASELINE_PORT) \
		D_NAME=voronoi2 D_URL=http://localhost:$(BASELINE_PORT)

ARENA_GAMES ?= 100
ARENA_SIMS ?= 1000
ARENA_ARGS ?=

arena-build: ## Build the head-to-head arena binary
	$(LIBTORCH_ENV) cargo build --release --manifest-path crates/snek-server/Cargo.toml --bin arena

arena: arena-build ## Play two nets head to head: make arena A=path B=path [ARENA_GAMES=100 ARENA_SIMS=1000 ARENA_ARGS="--parallel 8"]
	$(LIBTORCH_ENV) ./crates/snek-server/target/release/arena \
		--a $(A) --b $(B) --games $(ARENA_GAMES) --sims $(ARENA_SIMS) $(ARENA_ARGS)

clean: ## Remove build outputs
	cargo clean
	cargo clean --manifest-path crates/snek-train/Cargo.toml
	cargo clean --manifest-path crates/snek-server/Cargo.toml
