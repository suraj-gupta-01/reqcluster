#!/usr/bin/env bash
# GPU dev: CUDA torch + backend + frontend.
set -e
cd "$(dirname "$0")"
uv sync --extra all --extra gpu
exec uv run python scripts/dev.py
