.PHONY: dev test lint build up previews

dev:            ## API (embedded worker) + web dev server, no Docker
	./scripts/dev.sh

test:           ## all backend tests (integration tests download the Whisper model once) + web unit tests
	cd apps/api && .venv/bin/python -m pytest -q
	cd apps/web && npm test

lint:           ## linters and type checks
	cd apps/api && .venv/bin/ruff check . && .venv/bin/ruff format --check .
	cd apps/web && npm run lint && npm run typecheck

build:          ## production build of the web app
	cd apps/web && npm run build

up:             ## full stack with Docker Compose
	docker compose up --build

previews:       ## re-render the caption style preview images
	apps/api/.venv/bin/python scripts/render_style_previews.py
