# Amendment 2 (main thread) — applies on top of v1 + amendment 1

PREAMBLE additions (every spec):
- Portable: Windows + Linux, CPU + NVIDIA GPU. pathlib only; sqlite URLs built with Path.as_posix(); atomic writes via tmp file in same dir + os.replace (retry once on Windows PermissionError); threads only, never fork/multiprocessing; no shell-only scripts (Python entrypoints via [project.scripts]); open text files with encoding="utf-8" explicitly; newline="" for csv.
- Torch builds: `uv sync --extra cpu` or `--extra gpu` (cu126). Device auto: cuda > cpu; settings.embedding_device overrides.
- Tests never write outside tmp_path.

Design changes:
1. Store per-run X10 + 2D coords as `{home}/runs/{run_id}.npz` (np.savez_compressed, atomic). Refinement/active-learning load it; no UMAP recompute. Pruned with run retention.
2. JobRunner: two executors — `cpu` (max_workers=1: cluster, recluster, suggest, dependencies) and `io` (max_workers=settings.llm_concurrency: enrich). Both persist to jobs table; same 409 rules per (session, kind-group).
3. Spec merges to cut builder cold starts: S2+S4 (labels/sampling + pipeline), S5+S7 (metrics/analysis + refine), S10+S11 (gateway + tasks), S18+S19 (app + all routers), S21+S22 (pages A+B).
4. Acceptance suite (tests/acceptance) talks through an adapter fixture (LegacyApi -> NewApi). Vertical slice gate: S0,S1,S2+S4,S3,S9,S12,S13,S18 then run acceptance categories 1,2,3,6,7,8 against NewApi; must beat legacy baseline before continuing.
5. S27 infra: CI matrix [ubuntu-latest, windows-latest] for backend tests (cpu extra); start.sh + run-gpu.* replaced by `scripts/dev.py` (starts backend + frontend, cross-platform) — keep start.sh as a 2-line wrapper.
6. Defaults (embedder, reducer, HDBSCAN params) come from the experiment grid — fill in before S1.
