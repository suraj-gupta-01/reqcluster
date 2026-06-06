#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

echo "╔══════════════════════════════════╗"
echo "║         ReqCluster v5.0          ║"
echo "╚══════════════════════════════════╝"

# Backend
echo ""
echo "▶ Starting backend (FastAPI) on http://localhost:8000 ..."
cd "$ROOT/backend"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

# Frontend
echo "▶ Starting frontend (Vite) on http://localhost:5173 ..."
cd "$ROOT/frontend"
npm run dev &
FRONTEND_PID=$!

echo ""
echo "  Backend API  → http://localhost:8000"
echo "  API Docs     → http://localhost:8000/docs"
echo "  Frontend     → http://localhost:5173"
echo ""
echo "  Press Ctrl+C to stop both servers."
echo ""

trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT INT TERM
wait
