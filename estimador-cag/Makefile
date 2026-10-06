# Atajos de desarrollo del Estimador CAG. Uso: make <target> ("make help" para la lista).
# "make" sin argumentos equivale a "make run".
PORT ?= 8000

.PHONY: run test lint format live help

run: .env ## Arranca el servidor con recarga automática (PORT=8000 por defecto)
	uv sync --quiet
	@echo "Swagger: http://localhost:$(PORT)/docs"
	uv run uvicorn app.main:app --reload --port $(PORT)

test: ## Ejecuta los tests (LLM simulado, sin coste)
	uv run pytest

lint: ## Comprueba lint y formato con ruff
	uv run ruff check .
	uv run ruff format --check .

format: ## Formatea el código y aplica los arreglos automáticos de ruff
	uv run ruff format .
	uv run ruff check --fix .

live: ## Estimación real + evaluación (requiere el servidor en marcha)
	uv run python scripts/live_check.py --url http://localhost:$(PORT)

help: ## Muestra esta ayuda
	@grep -E '^[a-z]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-7s %s\n", $$1, $$2}'

.env:
	cp .env.example .env
	@echo "Creado .env a partir de .env.example: completa tu API key y vuelve a ejecutar make run."
	@exit 1
