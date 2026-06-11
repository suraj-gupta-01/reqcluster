# ReqCluster — User Manual

A practical guide to **running** ReqCluster in every mode (device, OS, database,
cache, LLM, deployment) and **using** every screen in the app.

For architecture, API reference, and performance numbers see [README.md](README.md).

---

## Contents
- [1. Prerequisites](#1-prerequisites)
- [2. Fastest start](#2-fastest-start)
- [3. Run modes & combinations](#3-run-modes--combinations)
  - [3.1 Compute: CPU vs GPU](#31-compute-cpu-vs-gpu)
  - [3.2 Database: SQLite vs PostgreSQL](#32-database-sqlite-vs-postgresql)
  - [3.3 Cache: file vs Redis](#33-cache-file-vs-redis)
  - [3.4 LLM: mock vs local vs cloud](#34-llm-mock-vs-local-vs-cloud)
  - [3.5 OS notes (Windows / Linux / macOS / WSL2)](#35-os-notes)
  - [3.6 Deployment: local vs Docker](#36-deployment-local-vs-docker)
  - [3.7 Ready-made recipes](#37-ready-made-recipes)
- [4. Environment variables](#4-environment-variables)
- [5. Using the app — every section](#5-using-the-app--every-section)
- [6. Troubleshooting / FAQ](#6-troubleshooting--faq)
- [7. Testing](#7-testing)

---

## 1. Prerequisites

| Need | Required? | Notes |
|---|---|---|
| **uv** (Python manager) | Yes | Runs the backend with Python 3.10; avoids the Windows torch DLL issue. |
| **Node.js 18+** | Yes | Frontend (Vite + React). |
| **Docker Desktop** | Optional | For the one-command full stack and for PostgreSQL/Redis. |
| **Ollama** | Optional | Only for **local LLM** mode. |
| **NVIDIA GPU + driver** | Optional | Only for GPU acceleration. Check with `python backend/core/device.py`. |

Everything has a safe default, so the app runs fully **offline on CPU with SQLite**
out of the box — no Docker, no GPU, no API keys.

---

## 2. Fastest start

Two terminals:

```bash
# Terminal 1 — backend (http://localhost:8000)
uv sync
cd backend
uv run uvicorn main:app --reload --port 8000

# Terminal 2 — frontend (http://localhost:5173)
cd frontend
npm install
npm run dev
```

Open **http://localhost:5173**, upload `data/aerospace_requirements.csv` or
`data/promise_requirements.csv`, and click **Run Clustering Pipeline**.

> First run downloads the SBERT model (~80 MB) once.

---

## 3. Run modes & combinations

ReqCluster has five independent "axes". Mix and match — pick one option per axis.

| Axis | Options | Controlled by |
|---|---|---|
| Compute | CPU (default) · GPU | which torch build is installed (auto-detected) |
| Database | SQLite (default) · PostgreSQL | `DATABASE_URL` |
| Cache | file `.npy` (default) · Redis | `REDIS_URL` |
| LLM | mock (default) · local · cloud | `REQCLUSTER_LLM_PROVIDER` |
| Deployment | local dev · Docker Compose | how you launch it |

### 3.1 Compute: CPU vs GPU

Embeddings run on the GPU automatically **when a CUDA build of torch is installed**.
Check what the runtime sees:

```bash
python backend/core/device.py          # human report
python backend/core/device.py --json   # machine-readable
python scripts/benchmark_embeddings.py --device cuda --n 10000   # measure
```

- **CPU (default):** nothing to do. ~440 requirements/sec embedding on a modern CPU.
- **GPU:** install a CUDA torch matching your driver, then just run normally:
  ```bash
  # check driver/CUDA, then (example for CUDA 12.1):
  pip install torch --index-url https://download.pytorch.org/whl/cu121
  ```
  ~8× faster embeddings (e.g., ~3,650 req/s on an RTX 2050).
- **GPU UMAP/HDBSCAN (cuML/RAPIDS):** Linux-only; on Windows use **WSL2**. The CPU
  path is always the default and fully functional.

> ⚠️ **Small-GPU + local LLM:** on a 4 GB GPU you cannot run GPU embeddings **and**
> a GPU-accelerated local LLM at once (they fight over VRAM → `CUDA error`). Either
> run embeddings on CPU and the LLM on GPU, or vice-versa.

### 3.2 Database: SQLite vs PostgreSQL

- **SQLite (default):** unset `DATABASE_URL`. Stored at `backend/data/reqcluster.db`.
  Best for a single local user. A startup banner reminds you; silence it with
  `REQCLUSTER_SQLITE_OK=1`.
- **PostgreSQL:** set `DATABASE_URL` (needs a running Postgres — easiest via Docker):
  ```bash
  docker compose up -d postgres
  # .env:
  DATABASE_URL=postgresql://reqcluster:reqcluster_dev_password@localhost:5432/reqcluster
  ```
  Use it for many users or > 20k requirements.

### 3.3 Cache: file vs Redis

The per-text embedding cache means re-runs and incremental additions skip
re-encoding.

- **File cache (default):** unset `REDIS_URL`. Embeddings cached as `.npy` on disk.
- **Redis:** set `REDIS_URL` (shared cache across processes/sessions):
  ```bash
  docker compose up -d redis
  # .env:
  REDIS_URL=redis://localhost:6379
  ```

> Redis vs file cache is **not** related to SQLite vs PostgreSQL — they are
> independent axes. For a single local user the file cache is just as fast.

### 3.4 LLM: mock vs local vs cloud

Set `REQCLUSTER_LLM_PROVIDER` in `.env`. It drives enrichment, refinement
rationales, and cluster summaries. Any provider failure falls back to deterministic
output, so the app never breaks.

- **`mock` (default):** deterministic, fully offline. No model needed.
- **`local` (Ollama / vLLM):**
  ```bash
  ollama serve                 # if not already running
  ollama pull qwen2.5-coder:7b # pick a model that fits free RAM
  # .env:
  REQCLUSTER_LLM_PROVIDER=local
  REQCLUSTER_LOCAL_LLM_URL=http://localhost:11434/api/generate
  REQCLUSTER_LOCAL_LLM_MODEL=qwen2.5-coder:7b
  REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS=120
  ```
  Model-size vs free RAM: ~1 GB → `qwen2.5:1.5b`, ~3 GB → `qwen2.5:3b`,
  ~6 GB → `qwen2.5-coder:7b`.
- **`openai` (any OpenAI-compatible gateway — Groq, OpenRouter, OpenAI):**
  ```bash
  # .env:
  REQCLUSTER_LLM_PROVIDER=openai
  REQCLUSTER_LLM_BASE_URL=https://api.groq.com/openai/v1
  REQCLUSTER_LLM_API_KEY=your_key
  REQCLUSTER_LLM_MODEL=llama-3.3-70b-versatile
  ```

After editing `.env`, **restart the backend** so it reloads.

### 3.5 OS notes

| OS | Notes |
|---|---|
| **Windows** | Use **PowerShell** and **`uv run`** for the backend (selects the right Python; avoids the `c10.dll` torch error). Venv paths use backslashes: `.venv\Scripts\python.exe`. |
| **Linux** | Native GPU clustering (cuML) is available with an NVIDIA GPU. Venv python at `.venv/bin/python`. |
| **macOS** | CPU path; Apple-Silicon GPU (MPS) is not used by the default model config. |
| **WSL2** | The way to get **GPU UMAP/HDBSCAN (cuML)** on Windows hardware: run the backend inside a WSL2 Linux distro with NVIDIA drivers + RAPIDS. |

### 3.6 Deployment: local vs Docker

- **Local dev:** the two-terminal flow in [§2](#2-fastest-start).
- **Docker Compose (full stack: postgres + redis + backend + frontend):**
  ```bash
  docker compose up --build      # open http://localhost:3000
  ```
  The backend image pre-downloads the SBERT model and has a healthcheck.

### 3.7 Ready-made recipes

**A. Lean local (recommended single-user) — fastest for one person**
SQLite + file cache + CPU + mock LLM. Just the [fastest start](#2-fastest-start). Zero setup.

**B. Local LLM mode** (private, on-prem AI)
```bash
ollama serve & ollama pull qwen2.5-coder:7b
# .env: REQCLUSTER_LLM_PROVIDER=local (+ the 3 local vars), REQCLUSTER_SQLITE_OK=1
cd backend && uv run uvicorn main:app --port 8000     # CPU embeddings (frees GPU for the model)
cd frontend && npm run dev
```

**C. GPU embeddings mode** (fastest clustering; LLM = mock or cloud)
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121   # into the backend venv
cd backend && uv run uvicorn main:app --port 8000     # embeddings auto-use the GPU
```

**D. Full production stack**
```bash
docker compose up --build     # postgres + redis + backend + frontend on :3000
```

---

## 4. Environment variables

All optional; copy `.env.example` → `.env`. The backend auto-loads `.env` at startup.

| Variable | Purpose | Default |
|---|---|---|
| `DATABASE_URL` | PostgreSQL DSN; unset = SQLite | unset (SQLite) |
| `REQCLUSTER_SQLITE_OK` | Silence the SQLite warning | unset |
| `REDIS_URL` | Redis embedding cache; unset = file cache | unset |
| `CORS_ORIGINS` | Allowed frontend origins (comma-sep) | `http://localhost:5173,http://localhost:3000` |
| `REQCLUSTER_LLM_PROVIDER` | `mock` / `local` / `openai` | `mock` |
| `REQCLUSTER_LOCAL_LLM_URL` | Ollama/vLLM generate endpoint | `http://localhost:11434/api/generate` |
| `REQCLUSTER_LOCAL_LLM_MODEL` | Local model name | `qwen2.5-coder:7b` |
| `REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS` | Local LLM timeout | `120` |
| `REQCLUSTER_LLM_BASE_URL` / `_API_KEY` / `_MODEL` | OpenAI-compatible gateway | unset |
| `REQCLUSTER_SMALL_N` | Adaptive small-vs-large pipeline threshold | `4000` |

---

## 5. Using the app — every section

The left rail is the **pipeline** (top group) and the **intelligence tools**
(bottom group). The top bar shows a live session-status chip (No session →
Processing → Ready).

> Many analysis screens need a **completed** clustering run. If you see
> *"No completed sessions found. Run clustering first."*, go to **Upload** and run
> the pipeline — that is expected, not an error.

### Pipeline

| Section | What it does | How to use |
|---|---|---|
| **Upload** | Imports a CSV/XLSX and runs the pipeline: SBERT embeddings → UMAP → HDBSCAN → c-TF-IDF labels → similarity graph. | Drag/drop a file, optionally open **Clustering Parameters** (min cluster size, min samples, similarity threshold), click **Run Clustering Pipeline**, watch progress, then **View Results**. |
| **Overview** | Dashboard summary of the result: totals, coverage, noise, and a card per cluster. | Click a cluster card to drill into **Cluster Detail**. |
| **Scatter Plot** | 2-D UMAP projection, colored by cluster (WebGL for large sets). | Hover points to inspect; zoom/pan; spot dense groups and outliers (noise). |
| **Similarity Graph** | Requirements as nodes, edges between semantically similar ones (ANN kNN). | Tune the similarity/edge filters; click nodes to inspect; find tightly-linked requirements. |
| **Dependency Tree** | Heuristic hierarchical / sequential / data / reference dependencies inferred from the text, with a generated rationale. 2-D and 3-D views. | Pick a session, **Generate dependencies**, then explore; click a node for its rationale; switch 2D/3D. |
| **Requirements** | The full, server-paginated, filterable table of requirements with their cluster assignment. | Search text, filter by cluster / noise, page through; the source of truth for raw items. |
| **Cluster Detail** | One cluster up close: label, keywords, size, member requirements, and representative examples. | Reached by clicking a cluster in Overview/Scatter. |

### Intelligence

| Section | What it does | How to use |
|---|---|---|
| **Enrichment** | LLM expansion of each requirement (intent, components, assumptions), and re-clustering on **base / enriched / hybrid** embeddings with comparison + ablation metrics. | Pick a session + mode, run enrichment; compare cluster quality across modes. Uses the configured LLM provider. |
| **Refinement** | **Phase 3 ClusterLLM.** Analyzes a completed clustering and proposes **merge** (clusters too similar) and **split** (cluster not coherent) suggestions, with coherence scores and rationales. | Pick a done session, **Generate suggestions**, review each card, **Accept/Reject** — every action is recorded in the audit log. *Requires a completed clustering session* (that's the "No completed sessions found" message). |
| **Review Queue** | **Phase 4 human-in-the-loop.** Pending manual cluster corrections that, when approved, become must-link / cannot-link constraints. | Approve/Reject pending items; the badge in the rail shows the pending count. |
| **Active Learning** | **Phase 5.** Surfaces the least-certain (uncertainty-sampled) assignments and folds your accepted corrections back into a **constrained re-cluster**; tracks quality across iterations. | Review low-confidence items, apply corrections, re-cluster with constraints, watch quality history. |
| **Export** | Download the result as a **PDF report** (metrics + charts + per-cluster requirements), **ReqIF 1.2**, **SysML/UML XMI**, **Jama** bundle, or **CSV**. | Pick a done session, click a format card to download. Dependency links are included when a dependency tree exists. |

---

## 6. Troubleshooting / FAQ

**"No clusters found / No completed sessions found. Run clustering first."**
The screen needs a finished clustering run. Go to **Upload → Run Clustering Pipeline**,
wait for *Ready*, then return.

**Too many requirements marked as noise (cluster -1).**
HDBSCAN's `min_cluster_size` (default `max(5, N/50)`) is too large for your data's
natural group size — common with very templated/boilerplate-heavy text. On the
**Upload** page open **Clustering Parameters** and lower **Min Cluster Size**
(e.g., 5-8), then re-run. Use **Refinement** afterward to merge over-split groups.

**Local LLM won't load / `requires more system memory`.**
The model needs more free RAM than is available. Close apps, or use a smaller model
(`qwen2.5:1.5b`/`3b`). Check loaded models with `ollama ps`.

**`CUDA error: shared object initialization failed` (Ollama).**
The backend is holding the GPU. On a small GPU, run the backend on CPU
(`uv run uvicorn ...`) so the local LLM can use the GPU — or vice-versa.

**Windows `c10.dll` / WinError 1114 when importing torch.**
Use **`uv run`** (it selects the project's Python). Avoid a stale `VIRTUAL_ENV`
pointing at another interpreter.

**SQLite warning at startup.**
Informational. It's the default local mode. Set `REQCLUSTER_SQLITE_OK=1` to silence,
or set `DATABASE_URL` to use PostgreSQL.

**Frontend can't reach the API.**
The backend must be on `:8000`; Vite proxies `/api` → `http://localhost:8000`.
Start the backend first.

---

## 7. Testing

```bash
uv run pytest -q                          # full backend suite (offline, deterministic)
uv run pytest --cov=backend               # with coverage
cd frontend && npm run build              # frontend production build
python scripts/benchmark_embeddings.py --n 10000   # embedding throughput on the detected device
```
