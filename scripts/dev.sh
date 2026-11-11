#!/usr/bin/env bash
# Start the whole stack for local development without Docker:
#   API with an embedded worker on :8000 and the Next.js dev server on :3000.
# Requirements: Python 3.11+, Node 20+, ffmpeg on PATH.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x apps/api/.venv/bin/python ]; then
  python3 -m venv apps/api/.venv
  apps/api/.venv/bin/pip install --upgrade pip
  apps/api/.venv/bin/pip install -e "apps/api[dev,llm]"
fi
if [ ! -d apps/web/node_modules ]; then
  (cd apps/web && npm install --no-audit --no-fund)
fi

export VIRALCUT_DATA_DIR="${VIRALCUT_DATA_DIR:-$PWD/data}"
export VIRALCUT_EMBEDDED_WORKER=true

(cd apps/api && exec .venv/bin/uvicorn viralcut.api.app:app --host 127.0.0.1 --port 8000) &
API_PID=$!
trap 'kill "$API_PID" 2>/dev/null || true' EXIT

cd apps/web
exec npm run dev
