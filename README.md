# ReqCluster — Phase 1 MVP

> Automatically group functionally related engineering requirements using SBERT embeddings, UMAP dimensionality reduction, HDBSCAN clustering, and c-TF-IDF labeling.

---

## Pipeline

```
CSV/XLSX Input
     │
     ▼
Preprocessing          (dedup · clean · validate)
     │
     ▼
SBERT Embeddings       (all-MiniLM-L6-v2 → 384-dim)
     │
     ▼
UMAP Reduction         (384-dim → 10-dim for clustering)
                       (384-dim → 2-dim  for visualization)
     │
     ▼
HDBSCAN Clustering     (density-based · noise-aware)
     │
     ▼
c-TF-IDF Labeling      (deterministic keyword extraction)
     │
     ▼
Similarity Graph       (cosine similarity > threshold)
     │
     ▼
Interactive Dashboard  (React + Plotly.js)
```

---

## Tech Stack

| Layer      | Technology                                      |
|------------|-------------------------------------------------|
| Backend    | Python 3.11, FastAPI, Uvicorn                   |
| ML         | sentence-transformers, umap-learn, hdbscan, scikit-learn |
| Database   | SQLite (via SQLAlchemy)                         |
| Frontend   | React 18, Vite, TailwindCSS, Plotly.js          |
| Container  | Docker + Docker Compose                         |

---

## Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node.js 18+
- pip

### 1. Clone and set up backend

```bash
git clone <repo>
cd reqcluster

# Install Python dependencies
pip install -r backend/requirements.txt

# Start the backend
cd backend
uvicorn main:app --reload --port 8000
```

### 2. Set up and start frontend

```bash
cd frontend
npm install
npm run dev
```

### 3. Open the app

- **Frontend**: http://localhost:5173
- **API Docs**: http://localhost:8000/docs

### One-command start (both servers)

```bash
chmod +x start.sh
./start.sh
```

---

## Docker (Recommended)

```bash
# Build and start both services
docker-compose up --build

# Access the app
open http://localhost:3000
```

---

## Input Format

Upload a **CSV** or **XLSX** file with these columns:

| Column    | Required | Description                     |
|-----------|----------|---------------------------------|
| `id`      | No       | Requirement identifier (e.g. REQ-001) |
| `text`    | **Yes**  | The requirement text            |
| `module`  | No       | Subsystem or domain             |
| `section` | No       | Section or chapter              |

**Example CSV:**
```csv
id,text,module,section
REQ-001,"Cooling fan shall activate above 70°C",Thermal,Temperature Control
REQ-002,"System shall survive 15g shock",Mechanical,Reliability
REQ-003,"Battery shall provide 8 hours runtime",Power,Endurance
```

A sample file is included at `data/sample_requirements.csv` (119 requirements across 6 domains).

---

## API Endpoints

| Method | Endpoint                          | Description                      |
|--------|-----------------------------------|----------------------------------|
| POST   | `/api/upload`                     | Upload CSV/XLSX file             |
| POST   | `/api/cluster`                    | Run clustering pipeline          |
| GET    | `/api/progress/{session_id}`      | Poll pipeline progress           |
| GET    | `/api/sessions`                   | List all sessions                |
| GET    | `/api/sessions/{id}`              | Get session details              |
| GET    | `/api/clusters?session_id=`       | Get all clusters                 |
| GET    | `/api/cluster/{id}?session_id=`   | Get cluster + its requirements   |
| GET    | `/api/graph?session_id=`          | Get similarity graph JSON        |
| GET    | `/api/requirements?session_id=`   | Get all requirements             |

Full interactive docs available at `/docs` (Swagger UI).

---

## Clustering Parameters

| Parameter             | Default        | Description                                     |
|-----------------------|----------------|-------------------------------------------------|
| `min_cluster_size`    | `max(5, N/50)` | Minimum requirements to form a cluster          |
| `min_samples`         | `3`            | HDBSCAN density sensitivity                     |
| `similarity_threshold`| `0.65`         | Cosine similarity cutoff for graph edges        |

Tune `min_cluster_size` smaller to capture more clusters (fewer noise points), or larger to get broader, more general clusters.

---

## Dashboard Pages

| Page              | Description                                           |
|-------------------|-------------------------------------------------------|
| **Upload**        | Drag-and-drop file upload with parameter configuration and live pipeline progress |
| **Overview**      | Summary stats, coverage, cluster list with sizes      |
| **Scatter Plot**  | 2D UMAP visualization · color by cluster · hover + click to inspect |
| **Similarity Graph** | Network graph of similar requirements · adjustable edge weight threshold |
| **Requirements**  | Full searchable/filterable/sortable table with membership scores |
| **Cluster Detail**| Per-cluster view with top keywords and ranked requirements |

---

## Project Structure

```
reqcluster/
├── backend/
│   ├── main.py              # FastAPI app entry point
│   ├── requirements.txt
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py        # All API endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── preprocessing.py # CSV/XLSX loading, dedup, validation
│   │   ├── embeddings.py    # SBERT embedding generation + cache
│   │   ├── reduction.py     # UMAP 384→10D and 384→2D
│   │   ├── clustering.py    # HDBSCAN clustering
│   │   ├── labeling.py      # c-TF-IDF keyword extraction + label generation
│   │   ├── graph.py         # Cosine similarity graph builder
│   │   └── pipeline.py      # Full pipeline orchestrator
│   └── models/
│       ├── __init__.py
│       ├── database.py      # SQLAlchemy models + SQLite setup
│       └── schemas.py       # Pydantic request/response schemas
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Router + sidebar layout
│   │   ├── main.jsx
│   │   ├── index.css         # Tailwind + custom components
│   │   ├── pages/
│   │   │   ├── UploadPage.jsx
│   │   │   ├── OverviewPage.jsx
│   │   │   ├── ScatterPage.jsx
│   │   │   ├── GraphPage.jsx
│   │   │   ├── ClusterDetailPage.jsx
│   │   │   └── RequirementsPage.jsx
│   │   └── utils/
│   │       ├── api.js        # Axios API client
│   │       └── colors.js     # Deterministic cluster color palette
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── package.json
├── data/
│   └── sample_requirements.csv
├── embeddings/              # Cached SBERT embeddings (.npy)
├── outputs/
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
└── start.sh
```

---

## Notes

- **Embedding cache**: Embeddings are cached by content hash in `embeddings/`. Re-uploading the same file skips re-encoding.
- **Noise cluster**: Requirements that don't fit any cluster are labeled cluster `-1` (Noise). Lower `min_cluster_size` to reduce noise.
- **Graph size**: For datasets > 500 requirements, the graph computes edges only for non-noise nodes (up to 500) to keep the visualization responsive. Max 2000 edges are stored.
- **No LLM**: All labeling is deterministic c-TF-IDF. No API keys required. Fully offline.

---

## Phase 1 Scope

✅ CSV/XLSX upload with validation  
✅ SBERT embeddings (all-MiniLM-L6-v2)  
✅ UMAP 384→10D (clustering) + 384→2D (visualization)  
✅ HDBSCAN clustering with noise detection  
✅ c-TF-IDF cluster labeling (deterministic)  
✅ Cosine similarity graph  
✅ Interactive Plotly scatter plot  
✅ Interactive network graph  
✅ Cluster detail view with membership scores  
✅ Full requirements table with search/filter/sort  
✅ Live pipeline progress  
✅ SQLite persistence  
✅ Docker support  

❌ LLM enrichment (Phase 2)  
❌ ClusterLLM refinement (Phase 2)  
❌ Active learning (Phase 3)  
❌ MBSE export (Phase 4)  
