.PHONY: install
install: ## Install the virtual environment and install the pre-commit hooks
	@echo "🚀 Creating virtual environment using uv"
	@git init
	@uv sync
	# @uv run pre-commit install

.PHONY: check
check: ## Run code quality tools.
	@echo "🚀 Checking lock file consistency with 'pyproject.toml'"
	@uv lock --locked
	@echo "🚀 Linting code: Running pre-commit"
	@uv run pre-commit run -a
	@echo "🚀 Static type checking: Running mypy"
	@uv run mypy

.PHONY: run-all
run-all: ## Run all data processing and modeling steps
	@echo "🌱 Data Acquisition"
	@uv run flow/data_acquisition.py
	@echo "🚀 Feature Engineering"
	@uv run flow/feature_engineering.py
	@echo "🎯 Predictive Modeling"
	@uv run flow/predictive_modeling.py
	@echo "💥 Squad Optimization"
	@uv run flow/squad_optimization.py

.PHONY: get-data
get-data: ## Get data
	@echo "🌱 Data Acquisition"
	@uv run flow/data_acquisition.py

.PHONY: features
features: ## Create features
	@echo "🚀 Feature Engineering"
	@uv run flow/feature_engineering.py
	
.PHONY: predictions
predictions: ## Train predictive model
	@echo "🎯 Predictive Modeling"
	@uv run flow/predictive_modeling.py

.PHONY: optimization
optimization: ## Run optimization
	@echo "💥 Squad Optimization"
	@uv run flow/squad_optimization.py

.PHONY: clean-data
clean-data: # Clean data folders
	@echo "💧 Clean data folders"
	@find data/* -type f -delete
	@touch data/datalog.json

.PHONY: test
test: ## Test the code with pytest
	@echo "🚀 Testing code: Running pytest"
	@uv run python -m pytest --doctest-modules

.PHONY: build
build: clean-build ## Build wheel file
	@echo "🚀 Creating wheel file"
	@uvx --from build pyproject-build --installer uv

.PHONY: clean-build
clean-build: ## Clean build artifacts
	@echo "🚀 Removing build artifacts"
	@uv run python -c "import shutil; import os; shutil.rmtree('dist') if os.path.exists('dist') else None"

.PHONY: docs-test
docs-test: ## Test if documentation can be built without warnings or errors
	@uv run mkdocs build -s

.PHONY: docs
docs: ## Build and serve the documentation
	@uv run mkdocs serve

.PHONY: help
help:
	@uv run python -c "import re; \
	[[print(f'\033[36m{m[0]:<20}\033[0m {m[1]}') for m in re.findall(r'^([a-zA-Z_-]+):.*?## (.*)$$', open(makefile).read(), re.M)] for makefile in ('$(MAKEFILE_LIST)').strip().split()]"

.DEFAULT_GOAL := help
