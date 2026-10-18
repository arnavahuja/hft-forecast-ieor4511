#!/usr/bin/env bash
# Start the FastAPI backend and the Vite front end together; Ctrl-C stops both.
set -euo pipefail
PYTHON="${PYTHON:-python}"
PNPM="${PNPM:-pnpm}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

"$PYTHON" -m hft_forecast.cli serve &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT INT TERM

if [ ! -d frontend/node_modules ]; then
  (cd frontend && "$PNPM" install)
fi
(cd frontend && "$PNPM" run dev)
