# Redesign working set

Resume point for the `overhaul` branch. Read in order: 01-plan, 02, 03, 04, 05 (later amendments override earlier ones). 04 sets the ML defaults from findings/experiments.md; 05 is the pre-build architecture review (contract fixes, SQLite embedding cache, seeded UMAP decision).
`findings/` = audit results (bugs by area) + profiling. Progress = `git log --oneline` (one commit per spec, message starts with the spec id, e.g. `S4:`).
Acceptance contract tests: `uv run pytest -m acceptance tests/acceptance` (new API by default; `REQCLUSTER_ACCEPTANCE_API=legacy` for the old one, baseline 53/104; slice categories 1,2,3,6,7,8 legacy 34/62).

## Status (2026-11-19, later)
Backend done: S0-S18 (+S19 folded into S14-S16), review fixes, enriched/hybrid modes. New API passes the acceptance suite 104/104 (legacy 53/104). Unit+api+legacy suites green.
Pipeline accuracy/speed check on real data: `uv run python docs/redesign/findings/exp/newpipe_gate.py <dir with emb_<dataset>.npy>` (ARI: uav2k .919, uav10k .925, multidomain .639, adversarial .404).
Next: frontend S20-S23, eval cutover S24 + S29, scripts S25, legacy delete S26, infra S27.
Do not `git push` until the user says so.
