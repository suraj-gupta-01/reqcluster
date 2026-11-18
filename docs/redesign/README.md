# Redesign working set

Resume point for the `overhaul` branch. Read in order: 01-plan, 02, 03, 04 (later amendments override earlier ones). 04 sets the ML defaults from findings/experiments.md.
`findings/` = audit results (bugs by area) + profiling. Progress = `git log --oneline` (one commit per spec, message starts with the spec id, e.g. `S4:`).
Acceptance contract tests: `uv run pytest -m acceptance tests/acceptance`.

## Status (2026-11-19)
Done: audits, profiling, experiment grid, datasets, acceptance suite (legacy baseline 53/104), deps on uv + Python 3.12 (`uv sync --extra all --extra cpu|gpu`).
Next: build the vertical slice (amendment 3, item 4) with the `builder` agent, then gate it on the acceptance suite. Delegate by agent role: `scout` haiku, `builder` sonnet, `architect` opus (see CLAUDE.md).
Do not `git push` until the user says so.
