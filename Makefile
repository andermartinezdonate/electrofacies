.PHONY: install install-app app train predict batch test lint clean info help

PYTHON ?= python
CONFIG ?= configs/default.yaml
INBOX  ?= data/wells/inbox
MODEL  ?= artifacts/latest
OUTPUT ?= outputs

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install package in editable mode with dev dependencies
	pip install -e ".[dev]"

install-app: ## Install package with Streamlit web app dependencies
	pip install -e ".[app]"

app: ## Launch the Streamlit web application
	streamlit run app.py

train: ## Train all model tiers and algorithms
	$(PYTHON) -m electrofacies.cli train --config $(CONFIG) -v

train-rf: ## Train only Random Forest models
	$(PYTHON) -m electrofacies.cli train --config $(CONFIG) --algorithms random_forest -v

train-xgb: ## Train only XGBoost models
	$(PYTHON) -m electrofacies.cli train --config $(CONFIG) --algorithms xgboost -v

predict: ## Predict a single well (set WELL=path/to/file.las)
	$(PYTHON) -m electrofacies.cli predict --well $(WELL) --model $(MODEL) --output $(OUTPUT) --config $(CONFIG) -v

batch: ## Batch predict all wells in inbox
	$(PYTHON) -m electrofacies.cli batch --inbox $(INBOX) --model $(MODEL) --output $(OUTPUT) --config $(CONFIG) -v

info: ## Show model artifacts info
	$(PYTHON) -m electrofacies.cli info --model $(MODEL)

test: ## Run test suite
	pytest tests/ -v --tb=short

lint: ## Run linter
	ruff check src/ tests/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-outputs: ## Remove all prediction outputs
	rm -rf outputs/*

clean-artifacts: ## Remove all model artifacts (CAREFUL!)
	@echo "This will delete ALL trained models. Press Ctrl+C to cancel."
	@sleep 3
	rm -rf artifacts/*
