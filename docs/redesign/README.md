# Redesign working set

Resume point for the `overhaul` branch. Read in order: 01-plan, 02, 03, 04, 05 (later amendments override earlier ones). 04 sets the ML defaults from findings/experiments.md; 05 is the pre-build architecture review (contract fixes, SQLite embedding cache, seeded UMAP decision).
`findings/` = audit results (bugs by area) + profiling. Progress = `git log --oneline` (one commit per spec, message starts with the spec id, e.g. `S4:`).
Acceptance contract tests: `uv run pytest -m acceptance tests/acceptance` (new API by default; `REQCLUSTER_ACCEPTANCE_API=legacy` for the old one, baseline 53/104; slice categories 1,2,3,6,7,8 legacy 34/62).

## Status (2026-11-20)
Everything is committed on local branch `overhaul`. The user will review it before any push.
- Tests: unit+api 298 pass; acceptance 104/104 (legacy was 53); hardening 68/68 on SQLite and on Postgres (`REQCLUSTER_TEST_DATABASE_URL`), plus the 35k scale test; eval fast suite all pass.
- Accuracy, ARI, MiniLM on CPU (gate `findings/exp/newpipe_gate.py`): uav2k .976-.991, uav10k .966, multidomain .664-.679, adversarial .32-.38.
- On an NVIDIA GPU, `auto` picks gte-modernbert: uav10k .998, multidomain .742.
- Optional `clusterer=ward` = ward groups + HDBSCAN noise.
- Speed (CPU): 2k job ~20 s; 10k ~1-3 min; 35k ~2-4 min.
- Read latency at 35k: <0.1 s for most endpoints, graph 0.33 s, CSV 0.9 s.
- Docker: 2.44 GB, offline; the first job in a fresh container takes 19 s at 2k.
- Real Ollama (`qwen3.5:4b`) works: about 17 s per requirement on a 4 GB GPU.

Decisions are recorded in 05 (sections A-K2) and 06 (second review).

Not verified: running on Windows (CI job exists, never run); the UI in a browser.
