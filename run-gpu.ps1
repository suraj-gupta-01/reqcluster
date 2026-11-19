# GPU dev: CUDA torch + backend + frontend.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
uv sync --extra all --extra gpu
uv run python scripts/dev.py
