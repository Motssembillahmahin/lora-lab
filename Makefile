# LoRA Lab — single entry point for all commands.
# Usage: `make help`

CONFIG ?= configs/qwen_0.5b_lora.yaml
ADAPTER ?= outputs/qwen-lora-adapter

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

.PHONY: setup
setup: ## Create the venv and install all dependencies via uv
	uv venv
	uv sync --extra dev
	@echo "Done. Activate with: source .venv/bin/activate"

.PHONY: train
train: ## Run a LoRA fine-tune (override with CONFIG=path)
	uv run python -m src.train $(CONFIG)

.PHONY: infer
infer: ## Chat with the fine-tuned adapter (override with ADAPTER=path)
	uv run python -m src.infer $(ADAPTER)

.PHONY: merge
merge: ## Merge the LoRA adapter into the base model
	uv run python -m src.merge $(ADAPTER)

.PHONY: lint
lint: ## Lint with ruff
	uv run ruff check src

.PHONY: format
format: ## Auto-format with ruff
	uv run ruff format src

.PHONY: test
test: ## Run the test suite
	uv run pytest -q

.PHONY: clean
clean: ## Remove caches and build artifacts (keeps outputs/ and data/)
	rm -rf .ruff_cache .pytest_cache **/__pycache__ build *.egg-info
