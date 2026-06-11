# Run ReqCluster on GPU (PowerShell):
#   PostgreSQL + Redis in Docker  +  backend with GPU embeddings  +  frontend.
#
# Usage (PowerShell, from the repo root):
#   .\run-gpu.ps1                 # LLM = mock (offline)
#   .\run-gpu.ps1 qwen2.5:3b      # also load a local Ollama model on the GPU
#
# If you get "running scripts is disabled", run once:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
param([string]$Model = "qwen2.5:3b")
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "==> 1/5 Ensuring GPU torch in the venv"
& $Py -c "import torch; assert torch.cuda.is_available()" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "    installing CUDA torch (one-time, ~2.5 GB)..."
  uv pip install --python $Py torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
}

Write-Host "==> 2/5 PostgreSQL + Redis (Docker)"
docker compose up -d postgres redis
Write-Host "    waiting for postgres..." -NoNewline
do {
  Start-Sleep -Seconds 1
  docker exec reqcluster-postgres pg_isready -U reqcluster -d reqcluster *> $null
} while ($LASTEXITCODE -ne 0)
Write-Host " ready"

# Datastore + cache (exported so they win over .env)
$env:DATABASE_URL = "postgresql://reqcluster:reqcluster_dev_password@localhost:5432/reqcluster"
$env:REDIS_URL    = "redis://localhost:6379"
$env:CORS_ORIGINS = "http://localhost:5173,http://localhost:3000"

if ($Model -eq "mock") {
  Write-Host "==> 3/5 LLM provider: mock (offline, deterministic)"
  $env:REQCLUSTER_LLM_PROVIDER = "mock"
} else {
  Write-Host "==> 3/5 Local LLM on GPU: $Model (Ollama)"
  ollama pull $Model
  $env:REQCLUSTER_LLM_PROVIDER     = "local"
  $env:REQCLUSTER_LOCAL_LLM_URL     = "http://localhost:11434/api/generate"
  $env:REQCLUSTER_LOCAL_LLM_MODEL   = $Model
  $env:REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS = "120"
}

Write-Host "==> 4/5 Backend (GPU embeddings) -> http://localhost:8000  (opens a window)"
Start-Process -FilePath $Py `
  -ArgumentList @("-m","uvicorn","main:app","--host","127.0.0.1","--port","8000") `
  -WorkingDirectory (Join-Path $Root "backend")

Write-Host "==> 5/5 Frontend -> http://localhost:5173  (opens a window)"
Start-Process -FilePath "cmd.exe" -ArgumentList @("/k","npm run dev") `
  -WorkingDirectory (Join-Path $Root "frontend")

Write-Host ""
Write-Host "  Backend + frontend are starting in their own windows."
Write-Host "  Opening http://localhost:5173 in ~8s ..."
Start-Sleep -Seconds 8
Start-Process "http://localhost:5173"
Write-Host "  Done. Close those two windows (or Ctrl+C in them) to stop the servers."
Write-Host "  Stop Docker:  docker compose stop postgres redis"
