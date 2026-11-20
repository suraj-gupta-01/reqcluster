#!/usr/bin/env bash
# One command to run ReqCluster locally: installs what is missing, then starts the backend,
# the frontend (and Ollama if REQCLUSTER_LLM_PROVIDER=local) and opens the browser.
# Torch build: NVIDIA GPU detected -> CUDA build, else CPU. Override: REQCLUSTER_TORCH=cpu|gpu.
set -e
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1; }
torch="${REQCLUSTER_TORCH:-}"
if [ -z "$torch" ]; then
  if command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then torch=gpu; else torch=cpu; fi
fi
uv sync --extra all --extra "$torch"
exec uv run --no-sync python scripts/dev.py "$@"
