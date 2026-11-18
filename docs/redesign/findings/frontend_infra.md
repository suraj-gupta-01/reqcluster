HIGH nginx.conf:2 no client_max_body_size -> 100m
HIGH Dockerfile.backend py3.11 + requirements.txt -> py3.12 + uv sync --frozen --no-dev --extra postgres --extra redis
HIGH ci.yml/eval.yml pip+requirements.txt -> setup-uv, py3.12; add branch triggers
HIGH run-gpu.sh:23/.ps1:14 .venv/Scripts on linux; torch pin vs cpu index revert
MED UploadPage.jsx:98 poll no unmount cleanup
MED UploadPage.jsx:98-117 poll before POST returns -> stale 'done'; catch no error state
MED ClusterDetailPage.jsx:57-74 no state reset on nav
MED RefinementPage.jsx:68, ActiveLearningPage.jsx:60 session-switch race; swallowed catch
MED App.jsx:239 sessionStatus stale on switch
MED eval/pipeline.py:94-110 implicit unpaged requirements
MED core/metrics.py:56-57,113 noise dropped -> inflated; hungarian 1-cluster 0.0
MED eval/deps_metrics.py:79-97 inferred_data non-whitelisted + dup
MED Dockerfile.frontend:5 npm ci lockfile
LOW eval/refine_metrics.py:96-110 mirrored dup suggestions recall>1
LOW eval/deps_metrics.py:107-113 empty expected recall=1.0
LOW eval/runner.py:62-66 engine created at import
LOW .env.example drift (SMALL_N unused; JAMA_*, SPACY_* undocumented)
LOW docker-compose exposed PG/Redis ports default pw; ./outputs
LOW start.sh:11 bare uvicorn
LOW run-gpu.ps1:28 stderr abort
LOW App.jsx no route lazy()
LOW ActiveLearningPage.jsx:103-106, DependencyTreePage.jsx:520 Plotly purge ref captured at mount
LOW api.js:79,87,95 manual URL concat; getErrorMessage strips ' at '
OK: api.js<->routes match; eslint deps warnings benign; no dangerouslySetInnerHTML
ARCH: process-local progress dicts, no job table; per-page session/fetch dup; infra not migrated
