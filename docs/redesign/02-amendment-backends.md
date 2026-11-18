# ReqCluster plan, amendment 1: swappable ML backends, Ollama gateway, startup model load

This amendment applies the profiling results and the Ollama requirements to the plan already sent. It replaces specs S1, S3, S4, S10 and S18, adds new spec S29, and edits S12 and S27. Everything else stands. I made no file edits.

## Decisions to add
- **Profiling.** At 10k rows on 12 cores, about 99% of the time is two steps: embedding (73 s cold) and UMAP (45-56 s). HDBSCAN takes 0.9 s, graph 1.4 s, the 2k dependency tree 0.58 s, and everything else under 0.1 s. Rust and numba are not justified, so no native code is planned.
- **Swappable backends, set by config.** Each has exactly one Protocol and no registry or factory layer: `get_embedder()` and `get_reducer()` are plain `match settings.x:` functions.
  - `Embedder` Protocol: `name`, `revision`, `is_mock`, `dim`, `encode(texts, batch_size) -> float32 ndarray`.
    - `SbertEmbedder(backend="torch"|"onnx", device)` wraps `SentenceTransformer(model, backend=..., device=...)`.
    - `HashingMockEmbedder` only when `embedding_backend="mock"`.
    - The ONNX backend needs the `onnx` extra. If the extra is missing, it raises Unavailable; there is no silent fallback.
  - `Reducer` Protocol: `fit_transform(X, seed) -> X10`. Two implementations: `UmapReducer` and `PcaReducer` (sklearn `PCA(n_components=reduce_dims, svd_solver="randomized", random_state=seed)`).
    - The default stays `umap` (legacy behaviour) until the S29 experiment decides.
    - The 2D view is always UMAP, fitted on the 10D clustering output rather than raw 384D. It is cheap and consistent with the clusters.
    - The 2D view shape changes compared with legacy. S29 reports this.
- **Model loads once, at startup.** The lifespan calls `get_embedder()` and does a 1-text warm-up encode. Startup fails fast if the model cannot load. Mock is instant, so tests and eval are unaffected.
- **One seed.** `settings.seed` goes to the reducer, UMAP 2D, the PCA solver, GMM and k-means in refine, and the hnswlib `random_seed`.
  - Trade-off: a seeded UMAP runs single-threaded. If `umap` stays the default, its 45-56 s will grow. S29 measures this.
- **Thread limits.** `settings.threads` (default `os.cpu_count()`) is applied once in the lifespan and at the start of each job:
  - `numba.set_num_threads` (guarded import)
  - `torch.set_num_threads`
  - `threadpoolctl.threadpool_limits(threads)` (threadpoolctl ships with sklearn, so no new dependency)
  - hnswlib `index.set_num_threads(threads)` inside `ml/graph.py`. The thread count is passed as an argument, so the ML layer stays free of env reads.

  With `job_workers=1`, jobs never oversubscribe each other.
- **Ollama local backend.**
  - Call: `POST {local_llm_url}/api/chat` with `{"model", "messages", "stream": false, "think": false, "format": <JSON Schema>, "options": {"temperature": 0, "seed": seed}}`. The answer is in `message.content`.
  - Structured calls: `Prompt.output_model` is a pydantic model. The gateway sends `output_model.model_json_schema()` as `format`, then parses with `model_validate_json`. A validation failure counts as a failed attempt, so it is retried, then falls back. No jsonschema dependency is needed.
  - Free-text calls (narratives) omit `format` but still send `think:false`.
  - The openai backend sends the same schema as `response_format: {"type": "json_schema", ...}`.
  - Default model is `qwen3.5:4b`. `gemma4:e4b` and `qwen3.5:2b` are documented alternates in `.env.example`. Default URL is `http://localhost:11434`.

## Target tree changes
```
  embeddings.py  Embedder Protocol; SbertEmbedder(torch|onnx, cpu|cuda|auto); HashingMockEmbedder; get_embedder() cached; embed()
  runtime.py     apply_thread_limits(settings), warm_up(settings); called by app lifespan and JobRunner worker start
  ml/reduce.py   Reducer Protocol; UmapReducer, PcaReducer; viz_2d(X10, seed) (replaces reduce part of ml/cluster.py)
  ml/cluster.py  HDBSCAN + auto min_cluster_size only
```

## S1: Config, errors, domain types (replaced)
Changes from the original S1:
- Add these Settings fields:
  - `embedding_backend` Literal `torch|onnx|mock`, default `torch`, env `REQCLUSTER_EMBEDDING_BACKEND`. This replaces the `embedding_model=="mock"` rule.
  - `embedding_device` Literal `auto|cpu|cuda`, default `auto`.
  - `reducer` Literal `umap|pca`, default `umap`.
  - `reduce_dims` default 10.
  - `threads` default `os.cpu_count()`, minimum 1.
- Change these defaults: `local_llm_url=http://localhost:11434` and `local_llm_model=qwen3.5:4b`.
- The test conftest (S0) sets `REQCLUSTER_EMBEDDING_BACKEND=mock` instead of `REQCLUSTER_EMBEDDING_MODEL=mock`.

Everything else in S1 is unchanged.

## S3: Embedder and store (replaced)
files: backend/reqcluster/{store.py,embeddings.py,runtime.py}, pyproject.toml, tests/unit/test_store.py, tests/unit/test_embeddings.py

change: Keep the store part of the original S3. Then:
- **Embedder.** Implement the Protocol, `SbertEmbedder` (torch|onnx) and `HashingMockEmbedder`.
  - `get_embedder()` is `lru_cache`d on settings.
  - Device `auto` means cuda if `torch.cuda.is_available()`, otherwise cpu. Port the adaptive batch size from legacy `core/embeddings.py`.
- **Store key.** Use `embedder.name` and `embedder.revision`. The backend is not part of the key: torch and ONNX outputs differ by about 1e-6, and sharing the cache is acceptable.
- **runtime.py.** `apply_thread_limits(settings)` sets numba, torch and threadpoolctl as described in Decisions. `warm_up(settings)` calls `get_embedder().encode(["warm-up"], 1)`.
- **pyproject.** Add the extra `onnx = ["sentence-transformers[onnx]"]` and add `onnx` to `all`.

callers: S13, S18 (lifespan)

accept: Previous tests, plus:
- `get_embedder()` returns the same object on a second call.
- `embedding_backend=onnx` without optimum installed raises Unavailable. Use a monkeypatched import to simulate the missing package.
- `apply_thread_limits` calls `numba.set_num_threads(n)` (fake module).

Then ALL.

depends: S1

parallel-with: S2 S9 S10 S12

## S4: Clustering pipeline (replaced)
files: backend/reqcluster/ml/{reduce.py,cluster.py,keywords.py,graph.py,pipeline.py}, tests/unit/test_pipeline.py

change:
- `reduce.py`:
  - Reducer Protocol.
  - `UmapReducer(dims)`: `n_neighbors=min(15, N-1)`; use `init="random"` when N<20; `random_state=seed`.
  - `PcaReducer(dims)`: dims capped at `min(dims, N-1, D)`.
  - `get_reducer(name, dims)` with a `match` on the name.
  - `viz_2d(X10, seed)`: UMAP to 2D.
- `cluster.py`: HDBSCAN only.
- `graph.build(..., threads, seed)` passes both to hnswlib.
- `pipeline.run(embeddings, texts, ids, params, reducer, progress=None)`. The reducer is passed in, so the pipeline stays pure.
- `ClusteringParams` gains `reducer` and `reduce_dims`, set from settings by the service.
- Edge cases are the same as the original S4.

callers: S13, S15 (refinement recomputes X10 through the same reducer and seed as the stored `run.params`)

accept: Previous tests, plus:
- Both reducers produce 2 clusters on the two-blob data.
- PCA dims are capped when N is small.
- Same seed gives identical X10 for both reducers.

Then ALL.

depends: S2

parallel-with: S5 S6 S7 S8

## S10: LLM gateway (replaced)
Changes from the original S10:
- `Prompt` gains `output_model: type[BaseModel] | None`.
- **Local backend (Ollama).** As described in Decisions: `/api/chat`, `stream:false`, `think:false`, `format` set to the schema when `output_model` is given, and `options` with temperature 0 and the seed.
- **openai backend.** `/chat/completions` with `response_format` json_schema when `output_model` is given.
- **Parsing.** `LLMResult.output` is an `output_model` instance, or a str for free text. A `ValidationError` counts as a failed attempt: retry, then breaker, then fallback.
- **Cache key** includes the provider, model and `output_model` schema hash.

accept: Previous tests, plus a fake `urlopen` asserting:
- The request body to `/api/chat` has `think` false and `format` equal to the schema.
- An invalid JSON response triggers a retry and then the fallback.
- A free-text call sends no `format`.

Then ALL.

## S12: Database, models, job runner (edit)
The worker thread's initialiser calls `runtime.apply_thread_limits(settings)`. Add `runtime.py` to callers. The S12 accept is unchanged. Because of this, S12 now depends on S3.

## S18: App and core routers (replaced)
Changes from the original S18:
- The lifespan order is: `apply_thread_limits` → `init_db` → `sweep_on_startup` → `warm_up`. If loading fails, the app exits with a clear log line.
- `/health` adds `{"embedder": name, "backend", "device", "reducer"}`.

accept: Previous tests, plus:
- `/health` reports `mock`.
- A lifespan with a failing embedder raises at startup (monkeypatched `get_embedder`).

## S27: Infrastructure (edit)
- `.env.example` documents `REQCLUSTER_EMBEDDING_BACKEND`, `REQCLUSTER_EMBEDDING_DEVICE`, `REQCLUSTER_REDUCER`, `REQCLUSTER_THREADS`, `REQCLUSTER_LOCAL_LLM_URL=http://localhost:11434` and `REQCLUSTER_LOCAL_LLM_MODEL=qwen3.5:4b` (alternates `gemma4:e4b`, `qwen3.5:2b`).
- The backend image builds with `--extra onnx` optional: a commented build arg `EXTRAS`.

## S29: Backend experiment harness (new)
files: eval/cli.py, eval/runner.py, eval/report.py, tests/eval/test_backends.py

change:
- Add CLI flags `--reducer umap|pca`, `--embedding-backend torch|onnx|mock` and `--threads N`. The runner maps them to env before `create_app`.
- The report records the backend, reducer, threads and per-step timings, taken from job progress timestamps or the `durations` in the clustering result.
- Add a `--matrix` mode that runs every reducer × backend pair on a dataset and writes one comparison table to `var/reports/`.

callers: `.github/workflows/eval.yml` (unchanged default run)

accept:
- `uv run python -m eval --help` lists the new flags.
- A test runs `--matrix` on 30 synthetic rows with the mock backend and both reducers, and asserts the report has 2 rows.

Then ALL.

depends: S24

parallel-with: S25 S26 S27 S28

**Updated parallel order.** S12 now waits for S3, since it imports `runtime.py`. Otherwise S12 still runs alongside S2, S9 and S10. S29 is the last ML-facing spec, and its result sets the `reducer` default in config.

I did not check whether the sentence-transformers version in `uv.lock` supports `backend="onnx"` (it needs ≥3.2). I also did not check whether `qwen3.5:4b`, `gemma4:e4b` and `qwen3.5:2b` exist as Ollama tags; I took them as given.

Risks:
- Fitting the 2D UMAP on the 10D output changes the scatter layout compared with legacy.
- If `umap` stays the default, seeding makes it single-threaded and slower than the measured 45-56 s.
- Loading the model at startup adds about 3-5 s to boot when the backend is not mock.