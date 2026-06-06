# ReqCluster — Scalable Pipeline Redesign (35k–50k requirements)

> Status: **IMPLEMENTED**. All UMAP pipeline optimizations, async background execution, read-only memory mapping, server-side pagination, and WebGL scatter plot visualization have been fully implemented and verified.

## Context

The current pipeline works well for the ~100–500 requirement demo but **will not
survive 35k–50k**. The dominant killer is UMAP pinned to a fixed `random_state`
(which forces single-threaded execution in `umap-learn`) and then run **twice**
(10-D + 2-D) on the full 384-D matrix — at 50k that alone is ~25–45 min or OOM.
On top of that, persistence is row-by-row ORM, the similarity graph is O(N²)
(capped to a meaningless first-500), graph nodes duplicate full requirement text
into a SQLite JSON blob, the `/cluster` request is held open for the entire run,
and the frontend loads **all** requirements into the browser.

Goal: a pipeline that ingests any size (50k+) and finishes in minutes on CPU and
under ~90s on GPU, with a UI that stays responsive, while preserving the
SBERT → reduce → HDBSCAN → c-TF-IDF → graph semantics.

Decisions (confirmed):
- **Hardware:** auto-detect — GPU path (RAPIDS cuML) when CUDA present, else an
  optimized parallel-CPU path.
- **Determinism:** adaptive — keep deterministic seeded path for small N
  (< ~4000), switch to fast parallel/approximate path for large N.
- **Scope:** backend pipeline + bulk persistence + async fire-and-poll job +
  serving (pagination, WebGL scatter, aggregated graph).
- **Dependencies:** may add ANN (`hnswlib`) and optional GPU (`cuml`) / ONNX
  (`onnxruntime`+`optimum`), gated so they're only imported where used.

---

## Honest review of the current pipeline (ranked by impact at 50k)

1. **UMAP single-threaded ×2 — the #1 bottleneck.** `reduction.py` sets
   `random_state=42` (disables `n_jobs` parallelism in umap-learn) and
   `low_memory=False`, then fits UMAP **twice** (10-D and 2-D) on full 384-D.
   The expensive kNN graph is computed twice. ~25–45 min / OOM at 50k.
2. **Embeddings: CPU-only, batch 64, whole-set cache.** No device auto-detect,
   no ONNX, and the cache is all-or-nothing (one `.npy` keyed by the hash of all
   texts joined) — any change re-encodes everything; no incremental.
3. **Row-by-row persistence.** `/cluster` mutates 50k ORM objects in a loop →
   50k UPDATEs on flush; upload does 50k `db.add()` inserts. Should be bulk.
4. **Similarity graph is O(N²) and unrepresentative.** `graph.py` does a dense
   `cosine_similarity` and, for N>500, only the **first 500** non-noise nodes —
   meaningless at 50k. Plus every node embeds the **full requirement text** into
   the stored JSON → huge DB blob and payload.
5. **Frontend loads everything.** `getRequirements(session_id)` returns all rows;
   RequirementsPage paginates **client-side**; ScatterPage uses SVG `scatter`
   (not WebGL) → browser death at 50k.
6. **`/cluster` blocks for the whole run.** It `await`s the threadpool to
   completion and holds the request open for minutes; the request's DB session is
   held the entire time.
7. **HDBSCAN `prediction_data=True` always** — extra memory/time not needed for
   batch clustering.
8. **Upload limits.** 25 MB cap (a 50k CSV can exceed it) and whole-file read
   into memory.
9. **Missing composite index** `(session_id, cluster_id)` for filtered queries.

### Other bugs / quick optimizations to fix along the way
- `prediction_data=True` → `False` (use `probabilities_` directly).
- Two UMAP fits recompute kNN twice → compute kNN once, reuse via
  `precomputed_knn`.
- Labeling `compute_ctfidf` calls `.toarray()` on the TF matrix — keep it sparse
  and take per-cluster top-k from sparse rows.
- Graph nodes should carry only `id, cluster_id, x, y, is_noise` (no text).
- Results dict keeps `embeddings`/`embeddings_10d` in memory after use — drop
  once persisted to cap RSS.

---

## New pipeline architecture

Adaptive backend with a hardware abstraction; each stage picks GPU vs CPU and
deterministic vs fast based on `N` and CUDA availability.

```
upload (stream, bulk insert)
  → embeddings        device auto-detect + adaptive batch + incremental cache → memmap on disk
  → PCA 384→50        fast, multi-threaded BLAS, denoise (sklearn TruncatedSVD/PCA)
  → kNN (once)        pynndescent / cuML; shared by both UMAP embeds
  → UMAP 50→10 + 50→2 reuse precomputed kNN; parallel (n_jobs=-1) or cuML; seeded only if N small
  → HDBSCAN(10-D)     boruvka_kdtree, core_dist_n_jobs=-1, prediction_data=False; or cuML HDBSCAN
  → c-TF-IDF          sparse, per-cluster top-k
  → ANN graph         hnswlib kNN over all nodes → lean node edges + cluster-level aggregate graph
  → bulk persist      bulk_update_mappings; embeddings to memmap; lean graph JSON
```

Run as an **async background job**: `/cluster` validates, sets status=processing,
enqueues, returns `202` immediately; client polls `/progress/{id}`.

### Stage detail & files

**A. Embeddings — `backend/core/embeddings.py` (+ new `backend/core/embedding_cache.py`)**
- `get_model()`: detect device (`torch.cuda.is_available()`), optional ONNX
  backend on CPU (`sentence-transformers` `backend="onnx"` via `optimum`).
- Adaptive `batch_size` (GPU 256–512, CPU 64–128); `encode(..., device=...)`.
- **Incremental content-addressed cache:** `EmbeddingCache` backed by an on-disk
  memmap matrix + a sidecar hash→row index (sqlite or json). Look up
  `sha256(text)`; encode only misses; assemble in input order. Re-runs &
  incremental adds become near-instant.
- Persist each session's matrix to `data/emb/session_<id>.f32.npy` (memmap), not
  the DB.

**B. Reduction — `backend/core/reduction.py`**
- PCA/TruncatedSVD 384→50 (fast, denoises, shrinks UMAP input).
- Compute kNN **once** (`umap.umap_.nearest_neighbors` / pynndescent), pass to
  both UMAP fits via `precomputed_knn=` → kNN paid once.
- Adaptive: `N < SMALL_N (~4000)` → `random_state=42` (deterministic);
  else `random_state=None`, `n_jobs=-1`, `low_memory=True`.
- GPU branch: `cuml.manifold.UMAP` when CUDA.

**C. Clustering — `backend/core/clustering.py`**
- `core_dist_n_jobs=-1`, `algorithm="boruvka_kdtree"`, `prediction_data=False`.
- GPU branch: `cuml.cluster.HDBSCAN` when CUDA.

**D. Labeling — `backend/core/labeling.py`**
- Keep c-TF-IDF; remove `.toarray()`; compute per-cluster top-k from sparse rows
  (`scipy.sparse` + `argpartition`).

**E. Graph — `backend/core/graph.py`**
- Build approximate kNN with **hnswlib** (cosine space) over all nodes; keep
  edges ≥ threshold, cap by weight (~10k). O(N log N).
- **Lean nodes:** `id, cluster_id, x, y, is_noise` only.
- Also build a tiny **cluster-aggregate graph** (clusters as nodes, summed
  inter-cluster similarity) for the default UI view; node-level edges on
  drill-down.

**F. Hardware abstraction — new `backend/core/compute.py`**
- `get_backend()` → `"gpu"` if cuML importable + CUDA, else `"cpu"`.
- Central `SMALL_N` threshold + adaptive knobs so all stages stay consistent.

### Persistence — `backend/api/routes.py`, `backend/models/database.py`
- Upload: `db.bulk_insert_mappings(Requirement, rows)` in one transaction.
- Cluster write: `db.bulk_update_mappings(Requirement, [...])` (id, cluster_id,
  membership_prob, umap_x, umap_y, is_noise).
- Add composite index `(session_id, cluster_id)` on `Requirement`.
- Store embeddings off-DB (memmap, path on `Session`); store **lean** graph JSON.

### Async job — `backend/api/routes.py` (+ small `backend/core/jobs.py`)
- `/cluster` → validate, set `processing`, submit to a process/thread worker
  (in-process `ThreadPoolExecutor` + the existing `pipeline_progress` dict and
  DB status), return `202 {session_id, status}`.
- Job opens its **own** DB session; bulk-persists; sets `done`/`error`.
- Keep `reset_stale_sessions()` on startup for crash recovery.
- Document Celery/RQ as the multi-node scale-out option.

### Serving — `backend/api/routes.py`, frontend pages
- `/requirements`: add `page`, `page_size`, `q`, `cluster_id`, `is_noise`;
  return `{items, total, page, page_size}` via SQL `LIMIT/OFFSET/WHERE`.
- New `/scatter-data?session_id=`: compact columnar `{id[], x[], y[], cluster_id[]}`
  (optionally downsampled for initial view); much smaller than full objects.
- New `/sessions/{id}/summary`: SQL-aggregated counts/coverage.
- Frontend: `ScatterPage` → Plotly **`scattergl`** (WebGL); `RequirementsPage` →
  server-side pagination/search; `GraphPage` → aggregate-first with drill-down.

### Dependencies
- Add `hnswlib` (base).
- Optional extras (gated imports, documented as `pip install` / `uv add` extras):
  `onnxruntime` + `optimum` (CPU embeddings), `cuml-cu12` / `cudf-cu12` (GPU).
- Frontend: none new (Plotly already ships `scattergl`).

---

## Performance targets (50k requirements)

| Stage | CPU (parallel) | GPU (cuML) |
|---|---|---|
| Embeddings (cold) | 2–5 min (ONNX faster) | 20–40 s |
| Embeddings (cached) | seconds | seconds |
| PCA + kNN + 2×UMAP-embed | 1–3 min | 10–30 s |
| HDBSCAN | 30–90 s | seconds |
| c-TF-IDF + ANN graph + persist | < 1 min | < 30 s |
| **Total** | **~4–8 min** | **< ~1.5 min** |

vs. current: single-threaded UMAP ×2 ≈ 25–45 min or OOM.
Complexity moves from O(N²) (graph) and serial-UMAP to ~O(N log N) throughout.

---

## Build order (incremental, each independently shippable & testable)

1. **Bulk persistence + bulk upload + composite index** (no algorithm change; immediate win, low risk).
2. **Reduction overhaul** — PCA pre-step, shared `precomputed_knn`, adaptive parallel/seed, `low_memory`. (Biggest single win.)
3. **HDBSCAN tuning** — `n_jobs`, boruvka, `prediction_data=False`.
4. **Embeddings** — device auto-detect + adaptive batch + incremental cache + off-DB memmap (+ optional ONNX).
5. **Graph** — hnswlib kNN + lean nodes + cluster-aggregate.
6. **Async job** — fire-and-poll `/cluster`.
7. **Serving** — paginated `/requirements`, `/scatter-data`, `/sessions/{id}/summary`; frontend scattergl + server pagination + aggregate graph.
8. **GPU backend** (`compute.py` + cuML branches) behind auto-detect.
9. **Labeling** sparse hardening.
10. **Benchmark + tests.**

---

## Verification

- **Synthetic benchmark:** generator for 1k / 10k / 50k requirements; a
  `scripts/benchmark_pipeline.py` that times each stage and asserts total under
  target and peak RSS under a ceiling.
- **Correctness at scale:** `n_clusters > 0`, `noise_rate < 15%`, silhouette on a
  sample ≥ 0.30, labels non-empty.
- **Incremental cache:** second run on the same/extended set is ≥ 5× faster.
- **Bulk persist:** 50k assignment write in a single transaction, < ~5 s.
- **ANN graph:** edges span all clusters (not first-500), all weights ≥ threshold,
  edge cap respected.
- **Async job:** `/cluster` returns `202` immediately; status transitions
  uploaded→processing→done; progress polled.
- **Serving:** `/requirements` returns a page (not 50k); `/scatter-data` payload
  size sane; ScatterPage renders 50k via WebGL without freezing.
- **Determinism gate:** small N reproducible across runs; large N stable cluster
  count (±small) across runs.
- Run full `uv run pytest` (offline mock) green; add new tests per stage.

## Risks / mitigations
- **cuML install is heavy / platform-specific** → keep it an optional extra,
  import-gated; CPU path is always the default and fully functional.
- **Approximate kNN/parallel UMAP → run-to-run variance** → bounded by the
  adaptive threshold (exact for small N); document it.
- **Incremental cache correctness** (hash collisions / stale vectors) → SHA-256
  keys + dimension/finite validation on load (pattern already in
  `domain_embeddings.py`).
- **Async job on a single worker** → fine for one node; Celery/RQ documented for
  horizontal scale.

---

## Candidate refinements (open for discussion, not yet decided)

- **Datastore at 50k:** SQLite is fine for reads but strained by concurrent
  writes + large JSON blobs + heavy pagination. Consider **Postgres** (or DuckDB
  for analytics) as the backbone, with a migration note.
- **Durable job infra:** the in-process `ThreadPoolExecutor` worker does not
  survive multi-worker uvicorn or restarts. Consider **Celery / RQ / Arq + a
  jobs table** for durable, multi-worker background processing.
- **Ultra-fast tier:** for very large N, optionally bypass UMAP entirely and
  cluster on PCA-reduced vectors (or MiniBatchKMeans) for maximum speed, trading
  some cluster quality — tiered by dataset size.
</content>
