PRODUCT REQUIREMENTS DOCUMENT                                                                                          
ReqCluster
AI-Powered Functional Requirement Clustering and Analysis Platform
Document Version
1.0
Status
Phase 1 Complete
Date
June 2025
Classification
Internal / Academic
Document Owner
ReqCluster Engineering Team
Review Cycle
Per Phase Milestone
Prepared by the ReqCluster Engineering Team
For Academic, Design-a-thon, and Industry Review Purposes
1. Project Overview
ReqCluster is an AI-assisted requirements engineering platform that automatically discovers, groups, and analyzes functionally related system requirements. The platform is designed to support systems engineers, business analysts, MBSE practitioners, and product managers who work with large-scale requirement repositories.
Requirements engineering is a foundational discipline in system development. However, as systems grow in complexity, requirement sets scale from dozens to thousands of entries, making manual organization, traceability, and review prohibitively expensive. ReqCluster addresses this gap by combining state-of-the-art sentence embedding technology, unsupervised machine learning, and interactive visualization into a cohesive, production-grade platform.
1.1 Core Value Proposition
Problem
ReqCluster Solution
Manual grouping of hundreds of requirements
Automated SBERT + HDBSCAN clustering pipeline
Hidden semantic dependencies between requirements
Cosine similarity graph with tunable edge threshold
Inconsistent cluster naming conventions
Deterministic c-TF-IDF keyword-based labeling
No traceability to system architecture
Future SysML / DOORS / Jama export (Phase 5)
Analyst subjectivity in cluster formation
Human-in-the-loop correction workflow (Phase 4)
1.2 Project Scope
The ReqCluster platform is structured as a phased delivery programme spanning two academic semesters. Phase 1 (the subject of this document's completion section) delivers the core ML clustering pipeline and interactive dashboard. Phases 2 through 5 extend the platform with LLM enrichment, cluster refinement, human feedback integration, active learning, and MBSE export capabilities.
2. Mission Statement
"
To eliminate the manual overhead of requirement organization by providing an intelligent, explainable, and human-supervised clustering platform that accelerates system design, improves traceability, and reduces review cycle time by at least 60%.
"
2.1 Design Principles
#
Principle
Implementation in ReqCluster
1
Explainability
All clusters labeled with c-TF-IDF keywords. No black-box outputs.
2
Human Oversight
Human-in-the-loop validation and correction integrated in Phase 4.
3
Open Source First
All components use open-source libraries. No vendor lock-in.
4
Determinism
Pipeline uses fixed random seeds (42) for reproducible outputs.
5
Incremental Delivery
Phased architecture ensures value delivery from Phase 1.
6
Scalability
Batch embedding, embedding cache, and graph edge limits ensure performance at scale.
3. Problem Statement
3.1 The Requirements Engineering Challenge
Modern engineered systems — aerospace vehicles, automotive platforms, medical devices, industrial control systems — routinely accumulate requirement repositories containing 500 to 50,000 individual requirements. These repositories are produced incrementally by multiple stakeholders across organizational boundaries, resulting in structurally heterogeneous, semantically redundant, and organizationally inconsistent requirement sets.
Requirements engineers currently spend an estimated 25–40% of their review cycle time on manual clustering, deduplication, and traceability mapping — tasks that are largely mechanical yet require semantic comprehension. This creates three compounding problems:
Latency: Requirement organization delays downstream design activities including interface control document (ICD) production and verification planning.
Subjectivity: Different analysts produce materially different cluster structures from identical requirement sets, reducing organizational consistency.
Incompleteness: Manual review misses indirect semantic relationships that automated similarity analysis captures reliably.
3.2 Quantified Problem Impact
Metric
Industry Baseline
ReqCluster Target
Manual clustering time (500 reqs)
8–12 hours per analyst
< 5 minutes automated
Cluster consistency across analysts
~60% agreement (Cohen's kappa)
> 90% deterministic
Hidden relationship detection
Dependent on analyst experience
Automated via similarity graph
Traceability to system elements
Manual, often incomplete
SysML export in Phase 5
3.3 Why Existing Tools Fall Short
Jama Connect and IBM DOORS provide requirement storage and versioning but offer no intelligent clustering capabilities.
Generic text clustering tools (k-means, LDA) do not leverage domain-specific semantic embeddings and require a priori cluster count specification.
LLM-based summarization tools (e.g., ChatGPT) require manual prompt engineering, produce non-deterministic outputs, and do not integrate with requirements management workflows.
ReqCluster uniquely combines domain-aware embedding (SBERT), topology-preserving projection (UMAP), density-based clustering (HDBSCAN), and deterministic labeling (c-TF-IDF) into an end-to-end automated pipeline.
4. System Architecture
4.1 Architectural Overview
ReqCluster follows a three-tier layered architecture separating data ingestion, ML processing, and application presentation. Each tier exposes clean interfaces enabling independent development and testing across team members.
SYSTEM ARCHITECTURE — THREE-TIER MODEL
TIER 3 — APPLICATION LAYER
React Dashboard  ·  FastAPI REST  ·  Plotly Visualization  ·  Similarity Graph UI
TIER 2 — ML PROCESSING LAYER
SBERT Embeddings  ·  UMAP Reduction  ·  HDBSCAN Clustering  ·  c-TF-IDF Labeling  ·  Similarity Graph Builder
TIER 1 — DATA LAYER
CSV Ingestion  ·  XLSX Ingestion  ·  SQLite Storage  ·  Embedding Cache  ·  Preprocessing Engine
4.2 Data Flow
Step
Component
Input
Output
Owner
1
Preprocessor
CSV / XLSX file
Cleaned DataFrame
Member 3
2
Embedding Engine
Requirement texts []
384-dim float32 array
Member 1
3
UMAP Reducer
384-dim embeddings
10-dim + 2-dim projections
Member 1
4
HDBSCAN Clusterer
10-dim projections
Cluster labels + probabilities
Member 1
5
c-TF-IDF Labeler
Texts grouped by cluster
Keywords + cluster labels
Member 2
6
Graph Builder
384-dim embeddings + 2D coords
Node + edge JSON
Member 1
7
FastAPI Layer
Pipeline results
REST responses + SQLite writes
Member 3
8
React Dashboard
REST API responses
Interactive visualizations
Member 4
4.3 Component Responsibilities
4.3.1 Data Layer
Component
Responsibility
preprocessing.py
CSV/XLSX loading with encoding detection, column name normalization, empty/duplicate removal, text cleaning, ID generation
database.py (SQLAlchemy)
Session, Requirement, Cluster, Graph ORM models; SQLite persistence; get_db() dependency injection
Embedding Cache
SHA-256 content hash keyed .npy files; skips re-encoding on duplicate upload
4.3.2 ML Processing Layer
Component
Responsibility
embeddings.py
Singleton SentenceTransformer loader; batch encoding (default 64); L2-normalized 384-dim outputs; cache read/write     
reduction.py
UMAP 384→10D (n_neighbors=15, min_dist=0.0, metric=cosine, seed=42); UMAP 384→2D (min_dist=0.1) for visualization      
clustering.py
HDBSCAN on 10D vectors; auto min_cluster_size=max(5,N/50); min_samples=3; prediction_data=True for membership probabilities
labeling.py
c-TF-IDF with bigram CountVectorizer; domain stopword list; top-5 keywords per cluster; deterministic label generation from top-4 keywords
graph.py
Cosine similarity matrix on original 384-dim embeddings; configurable threshold (default 0.65); node/edge JSON; max 2000 edges
pipeline.py
Orchestrator with step-by-step progress callbacks; wires all components in sequence; returns unified results dictionary
4.3.3 Application Layer
Component
Responsibility
routes.py (FastAPI)
9 REST endpoints; pipeline invocation; progress polling store; Pydantic validation; SQLAlchemy session management      
React App (Vite)
Client-side SPA with React Router v6; six pages; Tailwind CSS utility styling; dark theme with brand palette
Plotly.js Scatter
2D UMAP scatter plot; per-cluster color coding; hover templates; click-to-inspect side panel; cluster filter dropdown  
Plotly.js Graph
Similarity network visualization; adjustable edge weight slider; show-labels toggle; node click panel
5. Technology Stack
Layer
Technology
Version
Purpose
Runtime
Python
3.11+
Backend runtime environment
Web Framework
FastAPI + Uvicorn
0.111 / 0.30
Async REST API with OpenAPI docs
Embeddings
sentence-transformers
3.x
all-MiniLM-L6-v2 SBERT model
Dimensionality
umap-learn
0.5.x
Non-linear manifold projection
Clustering
hdbscan
0.8.x
Density-based noise-aware clustering
Labeling
scikit-learn
1.5+
CountVectorizer for c-TF-IDF computation
Data Processing
pandas + numpy
2.x / 1.26
DataFrame operations and array math
Database
SQLite + SQLAlchemy
2.0
Local persistent ORM-backed storage
Frontend
React 18 + Vite
18.3 / 5.x
SPA framework with HMR dev server
Styling
TailwindCSS
3.4
Utility-first dark-theme UI system
Visualization
Plotly.js
2.33
Interactive scatter + network graphs
HTTP Client
Axios
1.7
Frontend REST API communication
Container
Docker + Compose
—
Multi-service deployment orchestration
6. Detailed ML Pipeline Workflow
6.1 Step 1 — Requirement Ingestion and Preprocessing
The ingestion module accepts CSV and XLSX files. Column names are normalized against known variants (e.g., 'requirement', 'shall', 'content' all map to the canonical 'text' column). Text cleaning applies HTML tag removal, whitespace normalization, and quote stripping. Empty requirements (fewer than 5 characters after cleaning) and exact-text duplicates are removed before storage. Missing identifiers are auto-generated as REQ-NNN.
6.2 Step 2 — SBERT Embedding Generation
Requirements are encoded using the all-MiniLM-L6-v2 sentence transformer model, producing 384-dimensional L2-normalized float32 vectors per requirement. Batched processing (default batch_size=64) enables efficient GPU utilization when available. Embeddings are cached to disk using a SHA-256 content hash key, enabling instant replay on re-upload of identical requirement sets. The model captures semantic similarity beyond keyword overlap — semantically equivalent requirements expressed with different phrasing are positioned close in embedding space.
6.3 Step 3 — UMAP Dimensionality Reduction
Two separate UMAP projections are computed from the 384-dim embedding space:
10-dimensional projection (clustering): n_neighbors=15, min_dist=0.0, metric=cosine, random_state=42. The zero min_dist forces tight cluster formation suitable for HDBSCAN density estimation.
2-dimensional projection (visualization): n_neighbors=15, min_dist=0.1, metric=cosine. The non-zero min_dist preserves local structure while spreading clusters for visual clarity.
Both projections share the same random seed (42) ensuring reproducibility. n_neighbors is automatically bounded by dataset size to prevent errors on small requirement sets.
6.4 Step 4 — HDBSCAN Clustering
HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) operates on the 10-dimensional UMAP projection using Euclidean distance. Key parameter decisions:
min_cluster_size = max(5, N // 50): Auto-scaled to dataset size, preventing over-fragmentation on large sets.
min_samples = 3: Controls density threshold for core point designation.
prediction_data = True: Enables per-requirement cluster membership probability extraction.
cluster_selection_method = 'eom': Excess of Mass selection for stable cluster boundaries.
Requirements assigned cluster label -1 are designated as Noise — semantically isolated requirements that do not belong to any discovered cluster. Membership probabilities (0.0–1.0) indicate confidence of cluster assignment for non-noise requirements.
6.5 Step 5 — c-TF-IDF Cluster Labeling
Class-based TF-IDF (c-TF-IDF) treats each cluster's concatenated requirement texts as a single document class. A unigram+bigram CountVectorizer (max_features=5,000) extracts term frequencies. The IDF is computed as log(1 + N_clusters / (df + 1)) where df is the number of clusters containing a term. This rewards terms that are frequent within a cluster but rare across clusters — precisely the discriminative vocabulary that describes each cluster's functional domain. Terms matching a curated domain stopword list (including common requirements vocabulary such as 'shall', 'system', 'provide') are filtered. The top-4 remaining terms are concatenated to form the cluster label.
6.6 Step 6 — Similarity Graph Construction
A requirement-level similarity graph is constructed using cosine similarity on the original 384-dimensional embeddings (not the UMAP projections, which distort distances). An edge is created between any two requirements with cosine similarity exceeding the configured threshold (default 0.65). For datasets exceeding 500 non-noise requirements, edge computation is performed on the first 500 to maintain UI responsiveness. The total edge count is capped at 2,000 by weight (highest similarity edges retained). Node metadata includes requirement text, cluster assignment, and 2D UMAP coordinates for layout.
7. Phase 1 — Completed Implementation
STATUS: COMPLETE ✓   Phase 1 delivered and verified end-to-end.
7.1 Delivered Components
Component
Description
Status
CSV/XLSX Ingestion
Multi-format upload with column normalization and bad-line tolerance
✓ Complete
Requirement Preprocessing
Deduplication, empty removal, text cleaning, ID generation
✓ Complete
SBERT Embeddings
all-MiniLM-L6-v2 with batch processing and content-hash cache
✓ Complete
UMAP Reduction
Dual projection: 384→10D (clustering) + 384→2D (visualization)
✓ Complete
HDBSCAN Clustering
Density-based with noise detection and membership probabilities
✓ Complete
c-TF-IDF Labeling
Deterministic keyword extraction with domain stopword filtering
✓ Complete
Similarity Graph
Cosine-similarity edges with configurable threshold and cap
✓ Complete
FastAPI Backend
9 REST endpoints, Pydantic schemas, lifespan context manager
✓ Complete
SQLite Persistence
SQLAlchemy ORM: Session, Requirement, Cluster, Graph tables
✓ Complete
React Dashboard
6-page SPA: Upload, Overview, Scatter, Graph, Requirements, Cluster Detail
✓ Complete
Live Progress Polling
Step-by-step pipeline progress with 800ms polling interval
✓ Complete
Docker Support
Dockerfile.backend, Dockerfile.frontend, docker-compose.yml, nginx.conf
✓ Complete
7.2 API Endpoints — Phase 1
Method
Endpoint
Description
POST
/api/upload
Upload CSV/XLSX; returns session_id, requirement count, dedup stats
POST
/api/cluster
Trigger clustering pipeline with configurable parameters
GET
/api/progress/{session_id}
Poll pipeline progress: step name, percentage, message
GET
/api/sessions
List all upload sessions ordered by creation time
GET
/api/sessions/{id}
Retrieve single session with status and statistics
GET
/api/clusters?session_id=
Get all clusters ordered by size (descending)
GET
/api/cluster/{id}?session_id=
Get cluster details with full requirements list
GET
/api/graph?session_id=
Return similarity graph node+edge JSON
GET
/api/requirements?session_id=
Get requirements with optional cluster_id filter
7.3 Verified Performance — Sample Dataset
The pipeline was verified against a 118-requirement sample dataset spanning 6 functional domains (Thermal, Mechanical, Power, EMC, Safety, Communication). Pipeline performance on a standard development machine (Intel Core i7, 16GB RAM):  
Pipeline Stage
Execution Time
Output
Preprocessing
< 0.1 s
118 requirements (0 noise removed)
SBERT Embeddings
~8–15 s (first run)
118 × 384 float32 matrix
UMAP (10D + 2D)
~10–12 s
118 × 10 + 118 × 2 arrays
HDBSCAN
< 1 s
6 clusters, 0–5 noise points
c-TF-IDF Labeling
< 0.5 s
6 labeled clusters, 5 keywords each
Graph Construction
< 1 s
118 nodes, ~1100 edges at threshold 0.65
Total (cold start)
~30–35 s
Full pipeline with model download
Total (cached embeddings)
~12–15 s
UMAP + downstream only
8. Future Development Roadmap
The ReqCluster roadmap spans four future phases over eight weeks following Phase 1 completion. Each phase delivers production-ready features with measurable acceptance criteria.
Phase
Title
Duration
Primary Deliverable
Phase 1
Core ML Pipeline + Dashboard
Completed
SBERT + UMAP + HDBSCAN + React UI
Phase 2
LLM Semantic Enrichment
Weeks 1–2
Domain-aware embeddings + requirement expansion
Phase 3
ClusterLLM Refinement
Weeks 3–4
Cluster merge/split + representative extraction
Phase 4
Human-in-the-Loop System
Weeks 5–6
Manual correction + feedback workflow
Phase 5
Active Learning + MBSE Export
Weeks 7–8
Constraint-based retraining + SysML/DOORS/Jama export
8.1 Phase 2 — LLM Semantic Enrichment (Weeks 1–2)
Phase 2 enhances embedding quality by integrating LLM-generated semantic context. Requirements are expanded with domain-aware descriptions before embedding, improving clustering of sparse or ambiguous requirements. Domain-specific fine-tuning signals are injected to bias the embedding space toward engineering terminology.
Member
Duration
Tasks
Outputs
Owner Area
Phase 2
Wk 1–2
Member 1
Domain-aware embedding pipeline integration
Embedding comparison framework (cosine sim delta)
Ablation test: base vs. enriched embeddings
Performance profiling and batch optimization
Enhanced embedding module
Embedding comparison report
Benchmark dataset
Phase 2
Wk 1–2
Member 2
LLM prompt engineering for requirement expansion
Requirement expansion API (OpenAI / local LLM)
Domain vocabulary extraction module
Semantic augmentation quality evaluation
llm_enrichment.py service
Expansion quality metrics
Domain vocab dictionary
Phase 2
Wk 1–2
Member 3
LLM service API integration (async)
Enrichment result caching layer
API endpoint: POST /api/enrich
Database schema: EnrichedRequirement table
Enrichment API endpoints
Cache implementation
Updated DB schema
Phase 2
Wk 1–2
Member 4
Enrichment comparison UI panel
Before/after embedding visualization
Domain vocabulary tag display
Enrichment progress indicator
Enrichment comparison page
Updated scatter with toggle
Domain tag UI component
8.2 Phase 3 — ClusterLLM Refinement (Weeks 3–4)
Phase 3 applies ClusterLLM-inspired techniques to suggest cluster boundary corrections. An LLM evaluates each cluster's semantic coherence and proposes merge or split operations. Representative requirements (closest to cluster centroid) are extracted as human-reviewable cluster summaries.
Member
Duration
Tasks
Outputs
Owner Area
Phase 3
Wk 3–4
Member 1
Cluster merge candidate detection (silhouette analysis)
Cluster split candidate detection (bimodality test)
Centroid computation in embedding space
Boundary score computation
merge_suggest.py module
split_suggest.py module
Centroid extractor
Phase 3
Wk 3–4
Member 2
LLM coherence scoring prompt design
Merge/split rationale generation via LLM
Representative requirement extraction
Cluster summary paragraph generation
cluster_refinement.py
LLM rationale outputs
Representative req extractor
Phase 3
Wk 3–4
Member 3
Refinement suggestion storage (RefinementSuggestion table)
API: GET /api/suggestions, POST /api/apply-suggestion
Suggestion acceptance/rejection tracking
Audit log for applied refinements
Suggestion API layer
DB schema update
Audit trail implementation
Phase 3
Wk 3–4
Member 4
Merge/split suggestion review UI
Cluster comparison side-by-side panel
Representative requirement highlight
Suggestion acceptance workflow UX
Refinement review page
Suggestion card components
Accept/reject workflow UI
8.3 Phase 4 — Human-in-the-Loop System (Weeks 5–6)
Phase 4 delivers a structured human review workflow enabling domain experts to manually correct cluster assignments, capture feedback, and provide confidence ratings. Feedback is stored and used in Phase 5 active learning.
Member
Duration
Tasks
Outputs
Owner Area
Phase 4
Wk 5–6
Member 1
Constraint extraction from human feedback
Must-link / cannot-link pair identification
Feedback-to-embedding-space mapping
Constraint validation and conflict detection
Constraint extractor
Feedback-embedding bridge
Conflict detector
Phase 4
Wk 5–6
Member 2
LLM-assisted feedback interpretation
Confidence score generation from feedback
Natural language annotation processing
Cluster narrative update from corrections
Feedback interpreter
Confidence scorer
Narrative updater
Phase 4
Wk 5–6
Member 3
Feedback data model (Feedback, Correction, ConstraintPair)
API: POST /api/feedback, GET /api/review-queue
Review state machine (pending → reviewed → applied)
Feedback export (JSON, CSV)
Feedback API
Review queue system
Feedback export endpoint
Phase 4
Wk 5–6
Member 4
Requirement drag-and-drop cluster reassignment UI
Feedback capture form with confidence slider
Review queue dashboard with completion tracking
Annotation highlighting in scatter plot
Drag-and-drop cluster editor
Feedback form
Review queue page
Annotation overlay
8.4 Phase 5 — Active Learning + MBSE Export (Weeks 7–8)
Phase 5 closes the feedback loop with active learning and delivers MBSE integration. Must-link and cannot-link constraints from Phase 4 are injected into the clustering pipeline via constrained HDBSCAN. Export connectors for SysML, IBM DOORS, and Jama Connect complete the requirements management integration story.
Member
Duration
Tasks
Outputs
Owner Area
Phase 5
Wk 7–8
Member 1
Constrained HDBSCAN with must/cannot-link integration
Incremental re-embedding on new requirement addition
Active learning query strategy (uncertainty sampling)
Clustering quality delta tracking across iterations
Constrained clusterer
Incremental pipeline
Active learning module
Quality tracker
Phase 5
Wk 7–8
Member 2
SysML block diagram mapping from clusters
LLM-assisted requirement-to-element traceability
Cluster narrative finalization for export
Export summary document generation
SysML mapper
Traceability matrix
Export narrative generator
Phase 5
Wk 7–8
Member 3
DOORS ReqIF export connector
Jama Connect REST API integration
SysML XMI export serializer
API: GET /api/export/{format}
DOORS exporter
Jama connector
SysML XMI exporter
Export API
Phase 5
Wk 7–8
Member 4
Active learning uncertainty visualization
Export configuration and format selection UI
Constraint history and learning progress dashboard
Final analytics: coverage, noise reduction, timeline
Uncertainty viz
Export UI
Learning progress dashboard
Analytics panel
9. Team Work Division
The four-member team owns distinct subsystems. Phase 1 was a collaborative baseline delivery. Phases 2–5 assign clear primary ownership while maintaining shared integration responsibility.
Member
Role Title
Primary Subsystem
Phase 1 Contribution
Member 1
ML Infrastructure Lead
Embeddings · UMAP · HDBSCAN
core/embeddings.py, reduction.py, clustering.py, graph.py, pipeline.py
Member 2
LLM & Semantic Analysis Lead
Labeling · LLM Services
core/labeling.py (c-TF-IDF), cluster keyword generation
Member 3
Backend & Data Engineering Lead
FastAPI · Database · APIs
main.py, api/routes.py, models/database.py, models/schemas.py, preprocessing.py
Member 4
Frontend & UX Lead
React · Visualizations · UX
All 6 pages (Upload, Overview, Scatter, Graph, ClusterDetail, Requirements), App.jsx, utility modules
9.1 Cross-Cutting Responsibilities
Responsibility
Lead
Support
Architecture decisions
All members
Weekly sync required
Integration testing
Member 3
Members 1, 4 contribute test cases
Documentation
Member 3
Each member documents their own module
Performance profiling
Member 1
Member 3 instruments API latency
UX review
Member 4
All members participate in usability sessions
10. Repository Structure
reqcluster/
├── backend/                         # Member 3 — Backend & Data Engineering
│   ├── main.py                      # FastAPI app with lifespan context
│   ├── requirements.txt
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py                # All REST endpoint handlers
│   ├── core/                        # Member 1 — ML Infrastructure
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Ingestion + cleaning
│   │   ├── embeddings.py            # SBERT + cache
│   │   ├── reduction.py             # UMAP 10D + 2D
│   │   ├── clustering.py            # HDBSCAN
│   │   ├── labeling.py              # c-TF-IDF (Member 2)
│   │   ├── graph.py                 # Similarity graph
│   │   └── pipeline.py              # Orchestrator
│   └── models/
│       ├── database.py              # SQLAlchemy ORM models
│       └── schemas.py               # Pydantic I/O schemas
├── frontend/                        # Member 4 — Frontend & UX
│   ├── index.html
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── src/
│       ├── App.jsx                  # Router + sidebar layout
│       ├── main.jsx
│       ├── index.css
│       ├── pages/
│       │   ├── UploadPage.jsx
│       │   ├── OverviewPage.jsx
│       │   ├── ScatterPage.jsx
│       │   ├── GraphPage.jsx
│       │   ├── ClusterDetailPage.jsx
│       │   └── RequirementsPage.jsx
│       └── utils/
│           ├── api.js               # Axios client
│           └── colors.js            # Cluster palette
├── llm_services/                    # Member 2 — LLM & Semantic (Phase 2+)
│   ├── enrichment.py                # Requirement expansion
│   ├── refinement.py                # ClusterLLM merge/split
│   └── summarization.py             # Cluster narrative generation
├── active_learning/                 # Member 1 (Phase 5)
│   ├── constraints.py               # Must/cannot-link
│   └── retraining.py                # Incremental pipeline
├── export/                          # Member 3 (Phase 5)
│   ├── sysml_exporter.py            # XMI serializer
│   ├── doors_exporter.py            # ReqIF format
│   └── jama_connector.py            # REST API client
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── data/
│   └── sample_requirements.csv
├── embeddings/                      # Auto-generated cache
├── docs/
│   ├── PRD.docx
│   └── API_REFERENCE.md
├── docker-compose.yml
├── Dockerfile.backend
├── Dockerfile.frontend
├── nginx.conf
└── start.sh
11. Integration Contracts
11.1 Inter-Module Data Contracts
Producer
Consumer
Data Type
Schema / Format
preprocessing.py
embeddings.py
List[str]
Cleaned requirement texts, UTF-8
embeddings.py
reduction.py
np.ndarray
Shape (N, 384), float32, L2-normalized
reduction.py
clustering.py
np.ndarray
Shape (N, 10), float32, Euclidean space
reduction.py
graph.py + frontend
np.ndarray
Shape (N, 2), float32, viz coordinates
clustering.py
labeling.py
np.ndarray
Shape (N,), int32, values ≥ -1
labeling.py
routes.py
Dict[int, Dict]
{cluster_id: {label, keywords, size}}
graph.py
routes.py + Graph DB
Dict
{nodes: List[NodeDict], edges: List[EdgeDict]}
11.2 REST API Contract Summary
Endpoint
Request Schema
Response Schema
POST /api/upload
multipart/form-data: file (CSV/XLSX)
{session_id, filename, total_requirements, duplicates_removed, empty_removed, status}
POST /api/cluster
{session_id, min_cluster_size?, min_samples?, similarity_threshold?}
{session_id, total_clusters, noise_count, clusters[], status}
GET /api/graph
?session_id=int
{nodes: [{id, node_id, requirement_text, cluster_id, x, y, is_noise}], edges: [{source, target, weight}]}
12. Testing Strategy
12.1 Unit Testing
Module
Test Cases
Owner
preprocessing.py
UTF-8 / Latin-1 decode; column normalization; empty row removal; duplicate detection; malformed CSV tolerance
Member 3
embeddings.py
Embedding shape (N×384); L2 norm ≈ 1.0 per row; cache hit / cache miss behavior; batch boundary conditions
Member 1
reduction.py
UMAP output shapes (N×10, N×2); reproducibility with seed=42; n_neighbors auto-bound for small N
Member 1
clustering.py
Label range (≥ -1); probabilities in [0,1]; min_cluster_size auto-scaling; all-noise edge case
Member 1
labeling.py
c-TF-IDF keyword count per cluster; stopword filtering; label generation from single-keyword cluster; empty cluster handling
Member 2
graph.py
Edge count ≤ 2000 cap; threshold filtering; node count = N; large N truncation to 500
Member 1
12.2 Integration Testing
Test Scenario
Method
Pass Criterion
Full pipeline on sample_requirements.csv
POST /upload then POST /cluster
status='done', n_clusters ≥ 1, noise_count < N
Embedding cache retrieval
Upload identical file twice; compare timings
Second run < 50% time of first run
Empty file rejection
POST /upload with header-only CSV
HTTP 400, detail message present
Large file (1000 reqs)
Upload and cluster generated 1000-req CSV
Pipeline completes; edge count ≤ 2000
Similarity threshold filtering
GET /graph; vary threshold 0.5–0.95
Edge count monotonically decreases with threshold
Progress polling
GET /progress/{id} during active pipeline
step transitions: embedding→umap→clustering→labeling→graph→done
12.3 Acceptance Criteria
Metric
Target Value
Measurement Method
Silhouette Score (10D)
≥ 0.35
sklearn.metrics.silhouette_score on HDBSCAN labels
Noise requirement rate
< 15%
(noise_count / total_requirements) × 100
Human cluster approval rate
≥ 80%
3-analyst panel reviewing 50-req test set
API p95 latency (GET endpoints)
< 200ms
locust load test, 20 concurrent users
Dashboard initial render (FCP)
< 2s
Lighthouse audit on production build
Scatter plot render (200 pts)
< 1s
Plotly render timing in browser DevTools
Pipeline (500 reqs, cached emb)
< 60s
Wall-clock timing on reference hardware
Cluster label distinctiveness
0 duplicate labels per session
Programmatic check: len(set(labels)) == len(labels)
13. Success Metrics
13.1 Technical Metrics
Metric
Phase 1 Target
Phase 5 Target
Measurement
Clustering silhouette score
≥ 0.35
≥ 0.55 (with constraints)
sklearn silhouette_score
Noise rate
< 15%
< 5%
Noise count / total
API p95 latency
< 200ms
< 200ms
Load test (20 users)
Pipeline runtime (500 reqs)
< 60s
< 90s (LLM enrichment)
Wall-clock timing
Embedding cache hit rate
100% on repeat upload
100%
Cache log analysis
Test coverage
> 75%
> 85%
pytest-cov report
13.2 Business Metrics
Metric
Baseline (Manual)
ReqCluster Target
Improvement
Clustering time (500 reqs)
8–12 hours
< 5 minutes
99% reduction
Analyst consistency (Cohen's κ)
~0.60
1.0 (deterministic)
Full consistency
Hidden relationship detection
Analyst-dependent
Automated via graph
Systematic
Requirement review cycle time
~2 weeks (manual)
< 3 days
> 60% faster
MBSE traceability completeness
~40% (manual)
> 90% (Phase 5)
> 125% increase
13.3 User Metrics
Metric
Target
Collection Method
Human cluster approval rate
≥ 80%
3-analyst review of 50-req test set with structured rubric
Task completion rate (upload→cluster)
≥ 95%
Usability test: 10 first-time users
Time-to-first-cluster (new user)
< 5 minutes
Recorded usability session timing
System Usability Scale (SUS)
≥ 75 / 100
SUS questionnaire post-session
Review workflow completion (Phase 4)
< 30 min per 100 reqs
Timed review sessions with domain experts
Export adoption rate (Phase 5)
≥ 50% of sessions
API log analysis: export endpoint call rate
14. Future Enhancements
Beyond the Phase 2–5 roadmap, the following capabilities represent longer-term research and engineering opportunities for ReqCluster.
#
Enhancement
Description
Phase
1
Multi-Language Support
Use multilingual SBERT models (paraphrase-multilingual-MiniLM) to support requirements in German, French, Japanese, and Chinese.
Post-Phase 5
2
Real-time Collaboration
WebSocket-based multi-user review sessions with conflict detection and presence indicators.
Post-Phase 5
3
Ontology Integration
Connect to domain ontologies (NASA MBSE Ontology, AUTOSAR) for requirement-to-concept alignment.
Research
4
Requirement Quality Scoring
INCOSE-aligned quality assessment: measurability, verifiability, ambiguity detection via LLM.
Phase 6
5
Version Diffing
Cluster-level diff between requirement baseline versions to highlight structural changes over time.
Phase 6
6
Graph Neural Network Clustering
Replace HDBSCAN with a GNN-based approach that explicitly models requirement relationships as graph edges during clustering.
Research
7
Enterprise SSO + RBAC
SAML/OAuth2 authentication with role-based access control for enterprise deployment.
Phase 6
8
Compliance Mapping
Automatic mapping of clusters to regulatory standards (DO-178C, IEC 62443, ISO 26262 sections).
Research
15. References
[1]  Reimers, N. & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP 2019.    
[2]  McInnes, L., Healy, J. & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction. arXiv:1802.03426.
[3]  Campello, R.J.G.B., Moulavi, D. & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates. PAKDD 2013.
[4]  Grootendorst, M. (2022). BERTopic: Neural topic modelling with a class-based TF-IDF procedure. arXiv:2203.05794.  
[5]  Wang, Z. et al. (2023). ClusterLLM: Large Language Models as a Guide for Text Clustering. EMNLP 2023.
[6]  INCOSE. (2023). Systems Engineering Handbook, v5. International Council on Systems Engineering.
[7]  Object Management Group. (2019). Systems Modeling Language (SysML) v1.6 Specification.
[8]  IBM Engineering Requirements Management DOORS Next. IBM Corporation, 2024.
[9]  Jama Software. Jama Connect Product Documentation. jama.software, 2024.
[10]  FastAPI Documentation. Sebastián Ramírez. fastapi.tiangolo.com, 2024.
[11]  Plotly Technologies Inc. Plotly JavaScript Open Source Graphing Library. plot.ly, 2024.
[12]  Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825–2830.