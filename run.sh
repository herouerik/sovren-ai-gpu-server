#!/usr/bin/env bash
# Always run through .venv, regardless of what's active/on PATH.
# Root cause of past "unhashable type: dict" / random 500s: a bare `uvicorn`
# invocation resolved to ~/.local/bin/uvicorn (mismatched global starlette/jinja2)
# instead of this project's pinned .venv versions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p static data

# --reload's worker process can outlive a Ctrl-C that only reaches the
# reloader (see stop.sh) and keep holding the port -- clear it first, every
# time, so re-running this script is always a clean restart regardless of
# how the previous run actually died.
./stop.sh

exec .venv/bin/python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8082
