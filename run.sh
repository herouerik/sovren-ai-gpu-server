#!/usr/bin/env bash
# Always run through .venv, regardless of what's active/on PATH.
# Root cause of past "unhashable type: dict" / random 500s: a bare `uvicorn`
# invocation resolved to ~/.local/bin/uvicorn (mismatched global starlette/jinja2)
# instead of this project's pinned .venv versions.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mkdir -p static data
exec .venv/bin/python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8082
