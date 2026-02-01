# Makefile for Scoring Tools

.PHONY: setup test lint clean run docker-build docker-run help

PYTHON := uv run python
PYTEST := uv run pytest
RUFF := uv run ruff

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies using uv
	uv pip install -e .

test: ## Run tests with pytest
	$(PYTEST) tests/

lint: ## Run linting (ruff) - checks only
	$(RUFF) check .

format: ## Run formatting (ruff) - apply changes
	$(RUFF) check --fix .
	$(RUFF) format .

clean: ## Clean output directory and cache
	rm -rf output/
	rm -rf .pytest_cache
	rm -rf dist/
	find . -type d -name "__pycache__" -exec rm -rf {} +

run: ## Run the main pipeline (single segment)
	$(PYTHON) main.py

run-batch: ## Run batch processing for all segments
	$(PYTHON) run_batch.py

docker-build: ## Build Docker image
	docker build -t scoring-tools .

docker-run: ## Run Docker container (interactive)
	docker run -it --rm -v $(shell pwd)/data:/app/data -v $(shell pwd)/output:/app/output scoring-tools /bin/bash

# Configuration for default target
.DEFAULT_GOAL := help
