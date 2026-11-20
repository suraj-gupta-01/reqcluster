# ReqCluster

Upload a requirements file, get semantically grouped requirements you can inspect, correct and export. FastAPI backend, React frontend, SQLite by default.

## What it does

- **Cluster**: upload CSV/XLSX -> embed (sentence-transformers) -> UMAP -> HDBSCAN clusters with c-TF-IDF labels. Optional `ward` mode draws groups with Ward linkage and keeps HDBSCAN's outliers.
- **Explore**: 2-D UMAP scatter, kNN similarity graph, per-cluster detail, searchable and paged requirements table.
- **Dependency tree** with a rationale per edge; LLM edge explanations on demand.
- **LLM enrichment** and **refinement suggestions** (split/merge/move) behind a gateway: `mock`, OpenAI-compatible or local Ollama.
- **Feedback loop**: move a requirement, approve or reject, then constrained re-clustering. Active learning ranks the most uncertain items (margin) and tracks quality over time. Every change is audited.
- **Exports**: ReqIF, SysML XMI, Jama JSON, CSV, PDF.

## Architecture

### Components

```mermaid
flowchart LR
  UI["React (Vite)<br/>api/client.js, hooks useApi / useJob"] -->|"HTTP /api"| APP["FastAPI app.py<br/>API key, CORS, gzip"]
  APP --> API["api routers"]
  API --> SVC["services"]
  SVC --> ML["ml (pure numpy functions)"]
  SVC --> EMB["embeddings"]
  EMB --> CACHE[("SQLite embedding cache")]
  SVC --> LLM["llm gateway<br/>mock | openai | ollama"]
  LLM --- GW["retry, circuit breaker,<br/>response cache, in-flight caps"]
  SVC --> EXP["export<br/>reqif, xmi, jama, csv, pdf"]
  API --> JOBS["jobs.JobRunner<br/>thread pools, one active job per session"]
  JOBS --> SVC
  SVC --> DB[("DB: SQLite WAL (default) | Postgres")]
  JOBS --> DB
  SVC --> NPZ[("run arrays<br/>npz files in REQCLUSTER_HOME")]
```

`ml/` has no I/O: arrays in, arrays out. Services own persistence and orchestration. Routers validate input and return models.

### Clustering pipeline

```mermaid
flowchart TD
  A["upload: capped read, csv / xlsx,<br/>cleaning, dedupe"] --> B["embed<br/>auto: gte-modernbert on NVIDIA GPU, MiniLM on CPU<br/>per-text cache"]
  B --> C["exact BLAS kNN<br/>1k - 50k rows"]
  C --> D["UMAP 10-D cosine, seeded<br/>n_neighbors = min(80, max(15, n / 100)), 200 epochs"]
  D --> E{"clusterer"}
  E -->|"hdbscan (default)"| F["HDBSCAN"]
  E -->|ward| G["ward + HDBSCAN noise"]
  F --> H["c-TF-IDF labels"]
  G --> H
  H --> I["kNN similarity graph"]
  D --> J["2-D UMAP for the scatter<br/>(fed from the 10-D points)"]
```

### Job lifecycle

```mermaid
sequenceDiagram
  participant UI as Frontend useJob
  participant API as FastAPI
  participant R as JobRunner
  participant W as Worker thread
  participant DB as DB
  UI->>API: POST /api/sessions/{sid}/cluster
  API->>R: submit (one active job per session)
  R->>DB: insert job, status queued
  API-->>UI: 202 Job
  R->>W: run
  loop progress
    W->>DB: write progress
    UI->>API: GET /api/jobs/{id}
    API-->>UI: Job (status, progress)
  end
  W->>DB: succeeded or failed
  UI->>API: GET /api/jobs/{id}
  API-->>UI: terminal status, onDone fires
  Note over R,DB: POST /api/jobs/{id}/cancel sets cancelled.<br/>Startup sweep marks interrupted jobs failed.
```

### Data model

```mermaid
erDiagram
  sessions ||--o{ requirements : has
  sessions ||--o{ runs : has
  sessions ||--o{ enrichments : has
  sessions ||--o{ jobs : has
  runs ||--o{ assignments : has
  runs ||--o{ clusters : has
  runs ||--o{ suggestions : has
  runs ||--o{ feedback : has
  runs ||--o{ constraints : has
  runs ||--o{ audit : has
  runs ||--o{ dependency_trees : has
  runs ||--o{ dependency_explanations : has
  requirements ||--o{ assignments : "assigned in"
  runs {
    int revision
    int edit_seq
    int max_cluster_id
  }
  requirements {
    json attributes
  }
```

Run-scoped (a re-cluster starts clean, so stale data is never served): `assignments`, `clusters`, `suggestions`, `feedback`, `constraints`, `audit`, `dependency_trees`, `dependency_explanations`; a constrained re-cluster carries its constraints into the new run. Session-scoped (survive re-clustering): `requirements`, `enrichments`, `jobs`. A run records `revision`, `edit_seq` (manual edits) and `max_cluster_id` so edited cluster ids stay stable.

## Quick start

One command (needs [uv](https://docs.astral.sh/uv/) and Node.js 20+):

```bash
./start.sh        # Linux / macOS
powershell -ExecutionPolicy Bypass -File .\start.ps1   # Windows
```

It installs the Python deps (CUDA torch when an NVIDIA GPU is found, else CPU; override with
`REQCLUSTER_TORCH=cpu|gpu`) and the frontend deps on first run, starts Ollama if
`REQCLUSTER_LLM_PROVIDER=local`, starts the backend (:8000) and frontend (:5173), and opens the
browser. Ctrl+C stops everything it started. Copy `.env.example` to `.env` to change settings.

### Docker

```bash
docker compose up --build                # backend :8000, frontend :3000, SQLite in the reqcluster_var volume
export DATABASE_URL=postgresql://reqcluster:reqcluster_dev_password@postgres:5432/reqcluster
docker compose --profile full up         # adds Postgres (change the dev password)
```

The backend image is 2.44 GB and runs fully offline. It bakes MiniLM; for a GPU build pass `--build-arg EXTRAS="--extra gpu --extra fast --extra postgres"` and, to use another model, `--build-arg EMBED_MODEL=...`. The numba cache is baked in.

### Local LLM (Ollama)

```bash
ollama pull qwen3.5:4b
export REQCLUSTER_LLM_PROVIDER=local     # model: REQCLUSTER_LOCAL_LLM_MODEL=qwen3.5:4b
uv run python scripts/llm_smoke.py --help
```

Default provider is `mock` (deterministic, no network).

## Configuration

Read from the environment and `.env`; real env vars win. All variables with comments: [.env.example](.env.example).

| Variable | Default | Meaning |
|---|---|---|
| `REQCLUSTER_HOME` | `./var` | Data dir: DB, embedding cache, run arrays |
| `DATABASE_URL` | SQLite in HOME | Set `postgresql://...` for Postgres |
| `REQCLUSTER_API_KEY` | empty | Non-empty requires header `X-API-Key` |
| `CORS_ORIGINS` | localhost 5173, 3000 | Comma-separated |
| `REQCLUSTER_JOB_WORKERS` | 1 | Job threads (0 = inline) |
| `REQCLUSTER_SEED` | 42 | UMAP/HDBSCAN seed |
| `REQCLUSTER_EMBEDDING_MODEL` | `auto` | gte-modernbert on NVIDIA GPU, MiniLM on CPU, or a pinned model |
| `REQCLUSTER_EMBEDDING_DEVICE` | `auto` | `auto`, `cpu`, `cuda`. Pin `cpu` for reproducible clusters |
| `REQCLUSTER_EMBEDDING_BACKEND` | `torch` | `torch` or `mock` |
| `REQCLUSTER_EMBEDDING_REVISION` | 1 | Bump to invalidate the embedding cache |
| `REQCLUSTER_MAX_UPLOAD_MB` / `_MAX_ROWS` | 25 / 50000 | Upload caps |
| `REQCLUSTER_LLM_PROVIDER` | `mock` | `mock`, `openai`, `local` |
| `REQCLUSTER_LLM_BASE_URL`, `_API_KEY`, `_MODEL` | empty | OpenAI-compatible provider |
| `REQCLUSTER_LOCAL_LLM_URL` / `_MODEL` | `http://localhost:11434` / `qwen3.5:4b` | Ollama |
| `REQCLUSTER_LOCAL_LLM_CONCURRENCY` | 1 | Ollama serves one request at a time |
| `REQCLUSTER_LLM_MAX_RETRIES`, `_TIMEOUT_SECONDS`, `_CONCURRENCY` | 2, 30, 4 | Gateway resilience |
| `REQCLUSTER_BREAKER_THRESHOLD` / `_COOLDOWN_SECONDS` | 5 / 60 | Circuit breaker |
| `REQCLUSTER_JAMA_PROJECT_ID`, `_ITEM_TYPE_ID`, `_REL_TYPE_ID` | 0, 33, 1 | Jama export ids |

Cluster parameters (`POST /cluster` body): `clusterer` (`hdbscan` or `ward`), `min_cluster_size`, `min_samples`, `sim_threshold`, `top_k`, `embedding_mode`, `seed`, `enable_embedding_comparison`, `run_ablation`.

## API reference

All routes are under `/api`. Errors use the envelope `{"error": "<code>", "message": "<text>"}`. Long work returns `202` with a `Job`; poll `GET /api/jobs/{id}`. Interactive docs at `/docs`. The client contract lives in `frontend/src/api/endpoints.js`.

| Area | Method and path | Purpose |
|---|---|---|
| Sessions | `POST /sessions` | Upload file, create session |
| | `GET /sessions` | List sessions |
| | `GET /sessions/{sid}` | Session summary |
| | `DELETE /sessions/{sid}` | Delete session (204) |
| Clustering | `POST /sessions/{sid}/cluster` | Start clustering job (202) |
| | `GET /sessions/{sid}/clusters` | Clusters with labels |
| | `GET /sessions/{sid}/clusters/{cid}` | Cluster detail with its requirements |
| | `GET /sessions/{sid}/graph` | Scatter points and similarity edges |
| | `GET /sessions/{sid}/metrics` | Run metrics |
| | `GET /sessions/{sid}/requirements` | Paged list. Filters: `cluster_id`, `noise`, `search`; `sort`, `dir`, `page`, `page_size`. Total in `X-Total-Count` |
| Jobs | `GET /jobs/{job_id}` | Status and progress |
| | `POST /jobs/{job_id}/cancel` | Cancel |
| Feedback | `POST /sessions/{sid}/feedback` | Submit a move |
| | `GET /sessions/{sid}/feedback` | List feedback |
| | `GET /sessions/{sid}/feedback/export` | Export feedback |
| | `POST /sessions/{sid}/feedback/{fid}/approve` | Approve |
| | `POST /sessions/{sid}/feedback/{fid}/reject` | Reject |
| | `GET /sessions/{sid}/constraints` | Must/cannot-link constraints |
| | `POST /sessions/{sid}/recluster` | Constrained re-clustering job (202) |
| Active learning | `GET /sessions/{sid}/uncertainty?top_k=` | Most uncertain requirements |
| | `GET /sessions/{sid}/quality` | Quality over revisions |
| Refinement | `POST /sessions/{sid}/suggestions` | Generate suggestions job (202) |
| | `GET /sessions/{sid}/suggestions` | List |
| | `POST /sessions/{sid}/suggestions/{id}/accept` | Apply |
| | `POST /sessions/{sid}/suggestions/{id}/reject` | Dismiss |
| | `GET /sessions/{sid}/audit` | Audit trail |
| | `GET /sessions/{sid}/coherence` | Per-cluster coherence |
| Enrichment | `POST /sessions/{sid}/enrich` | LLM enrichment job (202) |
| | `GET /sessions/{sid}/enrichment` | Results |
| | `GET /sessions/{sid}/enrichment/summary` | Summary |
| Dependencies | `POST /sessions/{sid}/dependencies` | Build tree job (202) |
| | `GET /sessions/{sid}/dependencies` | Tree with rationale |
| | `POST /sessions/{sid}/dependencies/explain` | LLM explanation for an edge |
| Export | `GET /sessions/{sid}/export/{fmt}` | `fmt`: `reqif`, `sysml`, `jama`, `csv`, `pdf` |

## Metrics

All measured; methods and raw data are in [docs/redesign](docs/redesign/README.md).

### Accuracy

ARI (1.0 = perfect), mean of 3 seeds:

| Dataset | MiniLM (CPU default) | gte-modernbert (GPU default) | Ward + outliers (option) |
|---|---|---|---|
| UAV FMS 2k | 0.991 | 0.992 | 0.885 |
| UAV FMS 10k | 0.958 | 0.998 | 0.999 |
| Multidomain 1.2k | 0.679 | 0.742 | 0.722 |
| Adversarial 600 (20% junk) | 0.320 | 0.358 | 0.413 |

Small natural sets (64-100 rows) score 0.2-0.5 and are less stable.

Eval fast-suite baselines (MiniLM, CPU): uav_fms_2k ARI 0.975, purity 99.2%; multidomain 0.638; aerospace 0.474; bit_control 0.220; dependency_gold 0.175 (purity 87%); promise 0.110. Dependency reference recall 1.0 (aerospace 0.75).

Uncertainty ranking AUROC (margin): uav2k 0.932, multidomain 0.787, adversarial 0.861. Split/merge suggestions: false splits on clean gold groups fell from 2/5 to 0 after the silhouette gate.

### Speed

Dev laptop: 12-core CPU, RTX 2050 4 GB.

- Encode, texts/s. CPU: MiniLM 150, gte 17. GPU: MiniLM 3551, gte 863.
- Clustering job: 2k ~20 s, 10k ~1-3 min, 35k ~2-4 min (CPU).
- UMAP reduce at 2k: 3.9 s warm. The first call in a process adds ~20 s of numba JIT. A startup warm-up thread hides it; Docker bakes the cache (fresh-container 2k job: 19 s).
- 35k reduce: 118 s with exact BLAS kNN, 220 s with NN-descent.

API latency at 35k rows: requirements page 0.06-0.08 s, clusters 0.01 s, metrics 0.006 s, graph 0.33 s (3.25 MB raw, gzip level 1), CSV export 0.92 s, upload 1.4-2.5 s. Peak memory 1.07 GB.

LLM (local Ollama qwen3.5:4b, 4 GB GPU): ~17 s per requirement; cold model load ~100 s (the gateway loads once and uses keep_alive); edge explanation ~12 s.

### Quality gates

Tests: unit + api 298; acceptance 104/104 (legacy app 53/104); hardening 68/68 on SQLite and Postgres, plus 35k scale; eval fast suite passing. CI: Ubuntu + Windows matrix, Postgres job, frontend build and lint, eval workflow.

### Known limits

- Small uploads can cluster differently on CPU and GPU. Pin `REQCLUSTER_EMBEDDING_DEVICE=cpu` for reproducibility.
- Outlier recall on adversarial junk is ~21%.
- Windows has not been run yet (the CI job exists).

## Testing

```bash
uv run pytest -q                                    # unit + api (default)
uv run pytest -q -m acceptance                      # black-box contract suite
uv run pytest -q -m "hardening and not slow"        # fuzz, concurrency, failure injection
uv run pytest -q -m "hardening and slow"            # 35k scale
REQCLUSTER_TEST_DATABASE_URL=postgresql://... uv run pytest -q -m "hardening and not slow"   # Postgres
uv run ruff check backend tests
cd frontend && npm run build && npm run lint
```

Evaluation and tooling:

```bash
uv run python -m eval run --suite fast --model real          # gates on eval/baselines
uv run python -m eval --help                                 # run, update-baselines, drift, report
uv run python docs/redesign/findings/exp/newpipe_gate.py var/cache/gate   # accuracy gate
uv run python scripts/profile_graph.py pipeline --help       # profile the pipeline
uv run python scripts/bench_reads.py --help                  # read-path latency at scale
uv run python scripts/llm_smoke.py --help                    # enrich + deps + refine against the configured LLM
```

## Project structure

```
backend/reqcluster/
  app.py  config.py  db.py  models.py  domain.py  errors.py  runtime.py
  ingest.py  texts.py  embeddings.py  store.py  jobs.py
  api/        routers: sessions, jobs, feedback, refinement, enrichment, dependencies, export
  services/   clustering, sessions, assignments, feedback, active_learning,
              refinement, enrichment, dependencies
  ml/         pipeline, reduce, cluster, labels, graph, metrics, constraints, refine,
              active, analysis, keywords, sampling, dependency/   (no I/O)
  llm/        gateway, prompts, enrich, narrate
  export/     bundle (csv), reqif, xmi, jama, pdf
frontend/src/ api/ (client.js, endpoints.js)  hooks/ (useApi, useJob)  pages/  components/  state/
eval/         harness: datasets.yaml, baselines/, drift, report; run via python -m eval
scripts/      dev.py, llm_smoke.py, profile_graph.py, bench_reads.py, benchmark_embeddings.py
tests/        unit, api, acceptance, hardening, eval
docs/redesign/  design decisions, reviews, experiment findings
```

## Design decisions

The redesign plan, amendments, review outcomes and the experiments behind every default are in [docs/redesign/](docs/redesign/README.md). Start with the README there, then `05-amendment-review.md` and `06-review-2.md`.
