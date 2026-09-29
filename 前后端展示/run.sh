#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
exec python3 "$ROOT/backend/server.py" --host 0.0.0.0 --port "${PORT:-8765}"
