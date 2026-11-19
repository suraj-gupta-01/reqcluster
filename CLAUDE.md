# ReqCluster

FastAPI + React app: upload requirements -> embed (SBERT) -> UMAP -> HDBSCAN clusters -> labels, dependency tree, LLM enrichment/refinement, exports (ReqIF, SysML XMI, Jama, PDF).

## Commands

```bash
uv sync --extra all --extra cpu     # Python 3.12 env (.venv); use --extra gpu for CUDA 12.6 torch
uv run pytest -q                    # backend tests
uv run ruff check backend tests     # lint
cd frontend && npm install && npm run build && npm run lint
uv run python scripts/dev.py        # dev: backend :8000 + frontend :5173 (./start.sh wraps it)
uv run python -m eval --help        # evaluation harness
```

Optional extras (all lazily imported with fallbacks): `fast` hnswlib, `nlp` spaCy, `charts` matplotlib, `postgres`, `eval`.

## Code navigation: CodeGraph

`.codegraph/` index exists. Use it before grep/Read:
- `codegraph explore "<symbols or question>"`: source + call paths + blast radius in one call. Treat output as read.
- `codegraph query <name>`: locate a symbol. `codegraph node <name|file>`: one symbol/file with callers.
- After edits: `codegraph sync`. Fresh clone: `codegraph init`.

## Subagent routing

Delegate by job, cheapest capable model first. Agents live in `.claude/agents/`.

| Job | Agent | Model |
|---|---|---|
| search, locate, read, summarize files | `scout` | haiku |
| write/fix code from a spec, run tests | `builder` | sonnet |
| architecture, planning, hard root-cause, design review | `architect` | opus |

Rules:
- Plan first (architect or main thread), then hand builders self-contained specs: exact files, change, callers, accept command. A builder must not need to explore.
- Parallel builders only on disjoint files.
- Builders return a <=15-line receipt, not diffs. Verify with `git diff --stat` + tests, not by re-reading.

## Engineering rules

Karpathy guidelines are installed (`andrej-karpathy-skills` plugin): think before coding, simplicity first, surgical changes, goal-driven verification.
- Portable: must run on Windows and Linux, CPU and NVIDIA GPU. Use `pathlib` (no hard-coded `/` or `\`), `os.replace` for atomic writes, threads not `fork`, Python entrypoints (`uv run ...`) instead of shell-only scripts. Device is auto-detected (cuda > cpu) and overridable via config.
- Every bug fix ships with a test that failed before the fix.
- Keep the HTTP API contract in `backend/reqcluster/api/` in sync with `frontend/src/api/endpoints.js`.
- Never commit `.env`, `*.db`, model caches, or `.codegraph/`.
