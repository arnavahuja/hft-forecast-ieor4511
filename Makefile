# hft-forecast developer commands. Run inside the project conda environment
# (`conda activate backbone`), or override PYTHON=/path/to/python.
PYTHON ?= $(if $(CONDA_PREFIX),$(CONDA_PREFIX)/bin/python,python)
PNPM ?= pnpm
FRONTEND := frontend

.PHONY: dev install frontend-install test test-fast test-live cov lint fmt typecheck imports \
        frontend-lint frontend-test frontend-build e2e openapi check run api web clean

dev: install frontend-install  ## Install everything for development
	$(PYTHON) -m pre_commit install || true

install:  ## Install the Python package (editable) with dev extras
	$(PYTHON) -m pip install -e ".[dev]"

frontend-install:
	cd $(FRONTEND) && $(PNPM) install

test:  ## Python tests (no network)
	$(PYTHON) -m pytest -m "not live"

test-fast:
	$(PYTHON) -m pytest -m "not live and not slow" -x -q

test-live:  ## Opt-in tests that hit Yahoo / WRDS
	$(PYTHON) -m pytest -m live

cov:
	$(PYTHON) -m pytest -m "not live" --cov=hft_forecast --cov-report=term-missing

lint:
	$(PYTHON) -m ruff check backend tests user_plugins
	$(PYTHON) -m ruff format --check backend tests user_plugins

fmt:
	$(PYTHON) -m ruff format backend tests user_plugins
	$(PYTHON) -m ruff check --fix backend tests user_plugins

typecheck:
	$(PYTHON) -m mypy

imports:  ## Enforce layering rules
	$(dir $(PYTHON))lint-imports

frontend-lint:
	cd $(FRONTEND) && $(PNPM) run lint && $(PNPM) run format:check && $(PNPM) run typecheck

frontend-test:
	cd $(FRONTEND) && $(PNPM) run test

frontend-build:
	cd $(FRONTEND) && $(PNPM) run build

openapi:  ## Regenerate the OpenAPI schema and the TypeScript client types
	$(PYTHON) -m hft_forecast.cli openapi --out $(FRONTEND)/openapi.json
	cd $(FRONTEND) && $(PNPM) run gen:api && $(PNPM) exec prettier --write src/api/schema.d.ts

e2e:  ## Playwright smoke tests (starts API and front end)
	cd $(FRONTEND) && PYTHON=$(PYTHON) $(PNPM) run e2e

check: lint typecheck imports test  ## CI-style full check
	@if [ -d $(FRONTEND)/node_modules ]; then $(MAKE) frontend-lint frontend-test frontend-build; \
	else echo "frontend not installed; run 'make frontend-install'"; fi

run:  ## Start API (127.0.0.1:8000) and front end (127.0.0.1:5173) together
	PYTHON=$(PYTHON) PNPM=$(PNPM) ./scripts/run.sh

api:
	$(PYTHON) -m hft_forecast.cli serve

web:
	cd $(FRONTEND) && $(PNPM) run dev

clean:
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov .coverage
