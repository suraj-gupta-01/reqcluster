# ReqCluster

ReqCluster is an AI-assisted requirements clustering platform. It ingests CSV or XLSX requirement files, preprocesses and stores cleaned requirements, clusters them with SBERT + UMAP + HDBSCAN, labels clusters with c-TF-IDF, builds a similarity graph, and exposes the workflow through a FastAPI backend and React dashboard.

Phase 2 adds LLM semantic enrichment, Phase 3 introduces ClusterLLM automated refinement suggestion generation, and Phase 4 implements human-in-the-loop overrides with machine-learning constraints:

```text
Upload requirements
Preprocess and persist cleaned requirements
Run optional LLM enrichment for a session
Persist enriched requirement text and quality metadata
Cluster with base, enriched, or hybrid embeddings
Inspect comparison and ablation metrics
Review scatter, graph, requirements, overview, and cluster detail views
Generate automated merge/split suggestions (Phase 3)
Apply human-in-the-loop cluster overrides and generate ML constraints (Phase 4)
```

Base clustering remains the default Phase 1 behavior. Enriched and hybrid clustering require persisted enrichment from `POST /api/enrich`.

## Tech Stack

| Layer | Technology |
| --- | --- |
| Backend | Python 3.11, FastAPI, Uvicorn |
| ML | sentence-transformers, UMAP, HDBSCAN, scikit-learn |
| LLM enrichment | Mock provider, OpenAI-compatible HTTP provider, local HTTP provider |
| Database | SQLite via SQLAlchemy |
| Frontend | React, Vite, Tailwind CSS, Plotly.js |

## Local Setup

### 1. Backend

```bash
cd reqcluster
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

Backend URLs:

```text
API: http://localhost:8000
Swagger UI: http://localhost:8000/docs
Health: http://localhost:8000/health
```

### 2. Frontend

Open a second terminal:

```bash
cd reqcluster/frontend
npm install
npm run dev
```

Frontend URL:

```text
http://localhost:5173
```

The Vite dev server proxies `/api` requests to `http://localhost:8000`.

## Typical Phase 2 Workflow

1. Open `http://localhost:5173`.
2. Upload a CSV or XLSX file from the Upload page.
3. Run base clustering if you want the original Phase 1 flow.
4. Open the Enrichment page.
5. Select a session and provider. The default `mock` provider is deterministic and offline.
6. Click `Start Enrichment`.
7. Inspect enrichment status, enriched text, domain terms, confidence, and warnings.
8. Run `Hybrid` or `Enriched` clustering from the Enrichment page.
9. Inspect embedding comparison and ablation reports if enabled.
10. Open Scatter, Graph, Requirements, Overview, or Cluster Detail pages.

## Input Format

Upload a CSV or XLSX file with a requirement text column. Common text column names are normalized by the backend.

Recommended columns:

| Column | Required | Description |
| --- | --- | --- |
| `id` | No | Requirement identifier, for example `REQ-001` |
| `text` | Yes | Requirement text |
| `module` | No | Subsystem or domain |
| `section` | No | Document section |

Example:

```csv
id,text,module,section
REQ-001,"Cooling fan shall activate above 70C",Thermal,Temperature Control
REQ-002,"System shall survive 15g shock",Mechanical,Reliability
REQ-003,"Battery shall provide 8 hours runtime",Power,Endurance
```

## Core API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| POST | `/api/upload` | Upload and preprocess CSV/XLSX requirements |
| POST | `/api/enrich` | Enrich persisted requirements for a session |
| GET | `/api/enrich/status/{session_id}` | Get enrichment readiness counts |
| GET | `/api/enrich/results?session_id=` | Get persisted enrichment rows in requirement order |
| POST | `/api/cluster` | Run base, enriched, or hybrid clustering |
| GET | `/api/progress/{session_id}` | Poll clustering or enrichment progress |
| GET | `/api/sessions` | List sessions |
| GET | `/api/sessions/{session_id}` | Get one session |
| GET | `/api/clusters?session_id=` | Get clusters |
| GET | `/api/cluster/{cluster_id}?session_id=` | Get cluster details |
| GET | `/api/graph?session_id=` | Get similarity graph |
| GET | `/api/requirements?session_id=` | Get requirements |
| POST | `/api/suggestions/generate` | Generate merge/split refinement suggestions |
| GET | `/api/suggestions?session_id=` | List refinement suggestions for a session |
| POST | `/api/suggestions/apply` | Accept/reject a suggestion and log the audit entry |
| GET | `/api/suggestions/audit?session_id=` | List applied/rejected suggestion audit logs |
| POST | `/api/feedback/submit` | Submit human cluster reassignment and extract constraints |
| GET | `/api/feedback/queue` | Retrieve pending/reviewed human adjustments |
| POST | `/api/feedback/review` | Approve/reject adjustments (rejection rolls back) |
| GET | `/api/feedback/constraints` | Get active constraints and cycle/conflict validation report |
| GET | `/api/feedback/export` | Export review logs as CSV or JSON |


## Clustering Request Fields

```text
session_id: required positive integer
min_cluster_size: optional integer
min_samples: optional integer
similarity_threshold: 0..1, default 0.65
embedding_mode: base | enriched | hybrid, default base
enable_embedding_comparison: boolean, default false
run_ablation: boolean, default false
```

Public `enriched` and `hybrid` clustering do not silently fall back to base embeddings. Run `/api/enrich` first.

## LLM Provider Configuration

The mock provider requires no configuration.

OpenAI-compatible provider environment variables:

```text
REQCLUSTER_LLM_PROVIDER
REQCLUSTER_LLM_BASE_URL
REQCLUSTER_LLM_API_KEY
REQCLUSTER_LLM_MODEL
REQCLUSTER_LLM_TIMEOUT_SECONDS
REQCLUSTER_LLM_MAX_RETRIES
```

Local provider environment variables:

```text
REQCLUSTER_LOCAL_LLM_URL
REQCLUSTER_LOCAL_LLM_MODEL
REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS
```

Provider secrets stay in backend environment variables. The frontend does not accept API keys.

## Tests And Verification

Backend:

```bash
python -m pytest
```

Frontend:

```bash
cd frontend
npm run lint
npm run build
```

## Documentation

Phases 2, 3, and 4 details are documented in:

```text
docs/PHASE2_EMBEDDINGS.md
docs/PHASE2_LLM_ENRICHMENT.md
docs/PHASE2_ENRICHMENT_API_DB.md
docs/PHASE2_FRONTEND_ENRICHMENT_UI.md
docs/PHASE3_CLUSTER_REFINEMENT.md
docs/PHASE4_HUMAN_IN_THE_LOOP.md
```

## Notes

- Phase 1 base clustering is still the default and does not require enrichment.
- The mock enrichment provider is deterministic and suitable for offline tests.
- Enrichment file cache lives under `backend/cache/llm_enrichment/`.
- Embedding cache remains separate from enrichment cache.
- The backend currently persists the latest clustering coordinates per session, so true side-by-side persisted scatter comparison is a future backend enhancement.
