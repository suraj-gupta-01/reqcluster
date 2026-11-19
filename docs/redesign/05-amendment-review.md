# Amendment 5: architecture review before the vertical slice (2026-11-19)

Applies on top of 01-04. Later sections win. Verdict: the core design stands (in-process job runner with a jobs table, revision model, pure `ml/` layer, one LLM gateway, typed results, fetch-based frontend data layer). The changes below fix conflicts with the acceptance suite, remove code that buys nothing, and remove the seeded-UMAP speed penalty.

## A. Contract fixes (acceptance suite is the authority)
1. **Small N never errors.** Category 2 requires N=1..4 to finish with status done. Replace "N<5 raises InsufficientData" (S4, S13):
   - N < 5, or `min_cluster_size` > N: every point is noise, n_clusters=0, plus a warning. No UMAP.
   - 2D coordinates for tiny N: PCA to min(2, N-1) dims, zero-padded to 2 columns (all zeros when N=1).
   - `min_cluster_size` is clamped to [2, N]; `min_samples` to [1, N-1].
2. **One active job per session**, any kind. A second submit while one is queued or running returns 409 Conflict. This covers "enrich while clustering is 409" and "mutations during clustering are not 500". Mutating endpoints (feedback submit/approve, suggestion apply) also return 409 while a cluster/recluster job is active.
3. **Session read model** (`GET /api/sessions/{sid}`): `{id, name, created_at, n_requirements, status, total_clusters, noise_count, revision, last_job}`.
   - `status`: `uploaded` (no run, no job), `queued|running` (latest cluster-kind job active), `done` (current run exists and latest cluster-kind job succeeded), `error` (latest cluster-kind job failed).
   - `last_job` is the latest cluster-kind job, or null. It is what progress polling reads.
4. **Requirement read model**: `{id (int pk), req_id, ordinal, text, cluster_id (int, -1 noise), is_noise, probability, x, y}`. `id` order equals upload order.
5. Unknown `provider` values are 422 (pydantic Literal on the request model).

## B. Embedding cache: one SQLite file, no Redis (replaces store part of S3)
- `{home}/cache/embeddings.sqlite`, WAL, table `vec(key TEXT PRIMARY KEY, data BLOB)`; float32 bytes, dim known from the embedder.
- `get_many` queries in chunks of 900 keys; `put_many` uses `executemany` with `INSERT OR IGNORE` in one transaction. One `sqlite3` connection per call (thread safe, process safe with WAL).
- Why: the per-text `.npy` design writes 10k files for a 10k upload. File creation is slow on Windows (and antivirus scans each file). SQLite is one file, atomic, and works across uvicorn workers. It does not depend on `DATABASE_URL`, so Postgres deployments keep a local cache.
- Drop `RedisBackend`, `redis_url`, the `redis` extra and the Redis service in compose. A local cache on the same host gives the same hit rate without a network hop.

## C. UMAP: seeded and parallel (replaces the seed trade-off in 01, 02 and 04)
umap-learn 0.6.0 (installed) has a new layout pipeline: `compatibility_layout=False` with the Adam optimizer is reproducible with a fixed `random_state` and still runs in parallel. The legacy pipeline (`compatibility_layout=True`) forces a single thread when seeded. Decision is recorded in section F after the experiment.

## D. Persistence speed rules (every service)
- Bulk rows (assignments, clusters, graph edges, suggestions) are written with `session.execute(insert(Model), rows)`, never one ORM object per row.
- Per-run arrays (X10, 2D, probabilities) live in `{home}/runs/{run_id}.npz` (amendment 3 item 1). The DB keeps only what the API filters or sorts on.
- Graph edges are stored as one JSON blob per run in `graphs`, not one row per edge.

## E. Ownership changes for parallel builders
- `runtime.py` moves from S3 to S12 so S3 and S12 can run in parallel. `runtime.warm_up` calls `embeddings.get_embedder()` lazily.
- Services call only `reqcluster.embeddings.embed(...)`. `embed` looks up `get_embedder()` at call time, so tests and the acceptance adapter patch exactly one name: `reqcluster.embeddings.get_embedder`.
- Settings drop: `reducer`, `reduce_dims`, `embedding_backend=onnx`, `redis_url`. The pipeline is fixed by amendment 4 (PCA50, then UMAP 10D, then HDBSCAN).

## F. Experiment result (umap-learn 0.6.0, MiniLM, PCA50, UMAP 10D cosine nn=min(80,max(15,n//100)) min_dist 0 200 epochs, HDBSCAN n//50 s3 eom; 3 seeds; ARI / mean UMAP seconds)
| dataset | legacy unseeded | legacy seeded | adam seeded (random init) | adam seeded (spectral) |
|---|---|---|---|---|
| multidomain 1.2k | .627 / 0.7 | .637 / 2.0 | .669 / 0.6 | .656 / 0.7 |
| adversarial 600 | .343 / 0.3 | .330 / 0.7 | .329 / 0.3 | .340 / 0.3 |
| uav 2k | .929 / 1.5 | .936 / 3.2 | .904 / 1.2 | .885 / 1.2 |
| uav 10k | .909 / 16.2 | .904 / 15.7 | .864 / 4.6 | .899 / 5.2 |

All seeded variants reproduce labels exactly. Decision: `compatibility_layout=True` (set explicitly, silences the 0.6 FutureWarning), `random_state=seed`, `init="random"`. The seeded single-thread penalty is only ~1-2 s below 2k rows and absent at 10k, so determinism is free. Adam is 3x faster at 10k but costs up to 0.05 ARI on UAV; revisit only if UMAP time becomes the bottleneck after GPU encoding. Script: `umap06.py` in the session scratchpad (copied into findings/exp).
