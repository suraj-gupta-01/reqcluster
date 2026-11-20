# Amendment 7: third review (architect, 2026-11-20) and fix specs

Findings were reproduced against HEAD `dea0c4f`: #1 (`.env.example` parse), #2 (stale graph), #3 (thresholds), #14 (seed) and #15 (ingest). Each fix ships with a test that fails before it.

## Specs (builder-ready)

**S1. `.env` parsing.** Files: `config.py`, `test_config.py`.
- `_read_dotenv`: strip a trailing ` # comment` from unquoted values (regex `\s+#.*$`). Quoted values keep the content between the quotes.
- `get()`: treat an empty value as unset.
- `REQCLUSTER_SEED`: bounds 0 to 2**32-1.
- Test: `load_settings(env={}, dotenv=REPO/".env.example")` equals `load_settings(env={}, dotenv=None)` on every field except `home`. Today copying `.env.example` to `.env` fails with `REQCLUSTER_JOB_WORKERS: invalid value '1   # ...'`.
- Also: `.env.example` has `REQCLUSTER_HOME=./var`, which resolves against the working directory. Comment it out so the code default (absolute `<repo>/var`) applies.

**S2. Graph cache key and npz unlink.** File: `services/sessions.py`.
- `_graph_key` adds `run.created_at.isoformat()`. SQLite reuses the highest deleted ids, so a recreated session got the deleted session's cached graph (reproduced).
- `unlink_runs`: suppress `OSError` and log a warning unless it is `FileNotFoundError`. Today a Windows `PermissionError` during the post-commit prune marks a committed job failed.
- Tests: delete and recreate a session, then graph labels must match the new texts; an unlink that raises `PermissionError` must not fail `persist_run`.

**S3. Docker.** Files: `nginx.conf`, `Dockerfile.backend`.
- nginx `location /api`: `client_max_body_size 26m;`. The default is 1 MB, so the 1.16 MB uav_fms_10k upload got a 413.
- Dockerfile: `ARG EMBED_MODEL=all-MiniLM-L6-v2`; bake that model; runtime `ENV REQCLUSTER_EMBEDDING_MODEL=$EMBED_MODEL`. Today a GPU build resolves `auto` to gte, and offline startup fails.
- Move `HF_HOME` to `/app/hf-cache`, outside the `/app/var` volume, owned by `app`.
- Test (static): `client_max_body_size` in MB >= the `max_upload_mb` default.

**S4. Atomic job transitions.** File: `jobs.py`.
- `_execute`: `UPDATE ... SET status='running' WHERE id=? AND status='queued'`; if rowcount is 0, return. If the start write raises, mark the job failed (`WHERE status='queued'`).
- `cancel`: try queued -> cancelled conditionally first. If rowcount is 0, set `cancel_requested=True WHERE status='running'`.
- Test: a queued row (no worker), then `cancel`, then `_execute(job_id, fn)`: `fn` is never called and the status stays `cancelled`.

**S5. Ingest keeps inequalities.** File: `ingest.py`.
- `_TAGS = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(\s[^<>]*)?/?>")`.
- Test: `"Latency shall be < 10 ms when load > 50%."` stays unchanged (today it becomes `"Latency shall be 50%."`); `"<b>Pump</b> shall start"` becomes `"Pump shall start"`.

**S6. Seed bound.** File: `domain.py`.
- `ClusteringParams.seed: Field(None, ge=0, le=2**32-1)`. Today `-1` or `2**40` returns 202, then the job fails with an internal error.

**S7. Enrichment progress, cancel and summary.** Files: `llm/enrich.py`, `services/enrichment.py`, `OverviewPage.jsx`.
- `expand_all(..., on_chunk=None, chunk=32)` maps in chunks.
- The job calls progress and `check_cancel`, and persists after each chunk. Today a 2k local-LLM run (~9 h) shows no progress, cannot be cancelled and loses everything on restart.
- Summary: the latest row per requirement whatever its status, plus `failed` and `fallback` counts. Overview shows them and computes `max(size)` once.
- Tests: a cancel after chunk 1 leaves the job cancelled with chunk-1 rows saved; an openai provider without a base URL gives `total == n` and `fallback == n`.

**S8. Ollama reload after unload.** File: `llm/gateway.py`.
- On a local transport failure (URLError, timeout, OSError), set `_loaded = False` so the next attempt loads the model again.
- Test: `/api/chat` times out once, then `/api/generate` is called again and the result comes from the LLM.

**S9. Never cache fallback explanations.** File: `services/dependencies.py`.
- Persist only when the result is not a fallback, or the provider is `mock`.

**S10. Per-model calibration; runs record their model; ward cap.**
- Measured on multidomain: MiniLM gold coherence .393 vs random .199, and 2.2% of pairs have cosine >= .5. With gte: gold .659 vs random .558, and 83.5% of pairs reach .5. So the MiniLM thresholds label every gte random cluster "Highly coherent", and the graph and dependency filters stop working.
- `embeddings.CALIBRATION[model]` holds the coherence bands and the graph, merge and dependency thresholds. Measure the gte values with `refine_calibration.py`. For graph and dependency, use the same percentile of the random-pair cosine distribution as MiniLM's .5 and .45.
- `ClusteringParams.sim_threshold` and `SuggestParams.sim_threshold` default to None, resolved per model.
- Record `run.params["embedding_model"]`. `run_embeddings` warns when it differs from the current embedder.
- Ward: `linkage` builds an O(n²) `pdist` (4.9 GB at 35k, 10 GB at 50k). `clustering.start` rejects ward above 20000 requirements with `Invalid`.
- Fix the UploadPage text "ignored for ward" to say these settings drive outlier detection.

**S11. Eval pins the baseline model.** File: `eval/runner.py`.
- `REQCLUSTER_EMBEDDING_MODEL=all-MiniLM-L6-v2` and `REQCLUSTER_EMBEDDING_DEVICE=cpu`, unless already set.
- `_reset_backend` clears `get_embedder` and the gateway.

**S12. Windows launcher, README facts, CI triggers.**
- `start.ps1`: `if ($LASTEXITCODE) { exit $LASTEXITCODE }` after `uv sync`.
- README: `powershell -ExecutionPolicy Bypass -File .\start.ps1`.
- README: the 2-D UMAP is fed from x10, not from the embeddings.
- README: "Cluster detail with its requirements"; there are no representatives.
- README: the eval subcommand is `update-baselines`.
- README: a GPU Docker build uses `--build-arg EMBED_MODEL=...`.
- CI (`ci.yml`, `eval.yml`): push triggers `[main, overhaul]`.
