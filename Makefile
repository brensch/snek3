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
.PHONY: help test test-rust fmt lint train train-build frontend frontend-build api-build api clean

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

clean: ## Remove build outputs
	cargo clean
	cargo clean --manifest-path crates/snek-train/Cargo.toml
	cargo clean --manifest-path crates/snek-server/Cargo.toml
