#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Run ReqCluster on GPU (Windows + Git Bash):
#   PostgreSQL + Redis in Docker  +  backend with GPU embeddings  +  frontend.
#
# Usage (run inside Git Bash from the repo root):
#   ./run-gpu.sh                 # LLM = mock (offline, no VRAM used by an LLM)
#   ./run-gpu.sh qwen2.5:3b      # LLM = local Ollama model (pulled if missing)
#
# Good local models for a 4 GB GPU (lighter than qwen2.5-coder:7b):
#   qwen2.5:3b   (~2.0 GB)  best overall, same family
#   llama3.2:3b  (~2.0 GB)  strong general-purpose (Meta)
#   gemma2:2b    (~1.6 GB)  excellent for its size (Google)
#   qwen2.5:1.5b (~1.0 GB)  fastest, very capable, most headroom
#   phi3.5       (~2.2 GB)  strong reasoning (Microsoft)
# (SBERT on GPU uses only ~0.3 GB, so a 1.5b–3b model fits alongside it; Ollama
#  spills to CPU automatically if VRAM gets tight.)
# ─────────────────────────────────────────────────────────────────────────────
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
VENV_PY="$ROOT/.venv/Scripts/python.exe"
CURL="curl -s --noproxy *"

echo "==> 1/5  Ensuring GPU torch in the venv"
if ! "$VENV_PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
  echo "    installing CUDA torch (one-time, ~2.5 GB)..."
  uv pip install --python "$VENV_PY" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
fi
"$VENV_PY" "$ROOT/backend/core/device.py" | sed 's/^/    /'

echo "==> 2/5  PostgreSQL + Redis (Docker)"
docker compose up -d postgres redis
printf "    waiting for postgres"
until docker exec reqcluster-postgres pg_isready -U reqcluster -d reqcluster >/dev/null 2>&1; do printf .; sleep 1; done
echo " ready"

# Datastore + cache (exported, so they win over .env)
export DATABASE_URL="postgresql://reqcluster:reqcluster_dev_password@localhost:5432/reqcluster"
export REDIS_URL="redis://localhost:6379"
export CORS_ORIGINS="http://localhost:5173,http://localhost:3000"

# Optional local LLM
LLM_MODEL="${1:-}"
if [ -n "$LLM_MODEL" ]; then
  echo "==> 3/5  Local LLM: $LLM_MODEL (Ollama)"
  ollama pull "$LLM_MODEL"
  export REQCLUSTER_LLM_PROVIDER="local"
  export REQCLUSTER_LOCAL_LLM_URL="http://localhost:11434/api/generate"
  export REQCLUSTER_LOCAL_LLM_MODEL="$LLM_MODEL"
  export REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS="120"
else
  echo "==> 3/5  LLM provider: mock (offline)"
  export REQCLUSTER_LLM_PROVIDER="mock"
fi

echo "==> 4/5  Backend (GPU embeddings) on http://localhost:8000"
( cd backend && nohup "$VENV_PY" -m uvicorn main:app --host 127.0.0.1 --port 8000 \
    > "$ROOT/backend.log" 2>&1 & echo $! > "$ROOT/.backend.pid" )
printf "    waiting for backend"
until $CURL -o /dev/null http://127.0.0.1:8000/health >/dev/null 2>&1; do printf .; sleep 1; done
echo " ready"

echo "==> 5/5  Frontend on http://localhost:5173"
( cd frontend && nohup npm run dev > "$ROOT/frontend.log" 2>&1 & echo $! > "$ROOT/.frontend.pid" )
sleep 3

echo ""
echo "  ✅ ReqCluster is up:"
echo "     App :  http://localhost:5173"
echo "     API :  http://localhost:8000/docs"
echo "     logs:  backend.log , frontend.log"
echo "     stop:  kill \$(cat .backend.pid .frontend.pid) ; docker compose stop postgres redis"
