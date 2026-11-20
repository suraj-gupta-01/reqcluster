# One command to run ReqCluster locally on Windows: installs what is missing, then starts the
# backend, the frontend (and Ollama if REQCLUSTER_LLM_PROVIDER=local) and opens the browser.
# Torch build: NVIDIA GPU detected -> CUDA build, else CPU. Override: $env:REQCLUSTER_TORCH = "cpu"|"gpu".
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is required: https://docs.astral.sh/uv/getting-started/installation/"; exit 1
}
$torch = $env:REQCLUSTER_TORCH
if (-not $torch) {
    $torch = "cpu"
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
        & nvidia-smi *> $null
        if ($LASTEXITCODE -eq 0) { $torch = "gpu" }
    }
}
uv sync --extra all --extra $torch
if ($LASTEXITCODE) { exit $LASTEXITCODE }
uv run --no-sync python scripts/dev.py @args
