# Redesign working set

Resume point for the `overhaul` branch. Read in order: 01-plan, 02, 03, 04, 05 (later amendments override earlier ones). 04 sets the ML defaults from findings/experiments.md; 05 is the pre-build architecture review (contract fixes, SQLite embedding cache, seeded UMAP decision).
`findings/` = audit results (bugs by area) + profiling. Progress = `git log --oneline` (one commit per spec, message starts with the spec id, e.g. `S4:`).
Acceptance contract tests: `uv run pytest -m acceptance tests/acceptance` (new API by default; `REQCLUSTER_ACCEPTANCE_API=legacy` for the old one, baseline 53/104; slice categories 1,2,3,6,7,8 legacy 34/62).

## Status (2026-11-19)
Done: audits, profiling, experiment grid, datasets, acceptance suite (legacy baseline 53/104), deps on uv + Python 3.12 (`uv sync --extra all --extra cpu|gpu`).
Next: build the vertical slice (amendment 3, item 4) with the `builder` agent, then gate it on the acceptance suite. Delegate by agent role: `scout` haiku, `builder` sonnet, `architect` opus (see CLAUDE.md).
Do not `git push` until the user says so.
