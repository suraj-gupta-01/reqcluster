# Redesign working set

Resume point for the `overhaul` branch. Read in order: 01-plan, 02, 03, 04, 05 (later amendments override earlier ones). 04 sets the ML defaults from findings/experiments.md; 05 is the pre-build architecture review (contract fixes, SQLite embedding cache, seeded UMAP decision).
`findings/` = audit results (bugs by area) + profiling. Progress = `git log --oneline` (one commit per spec, message starts with the spec id, e.g. `S4:`).
Acceptance contract tests: `uv run pytest -m acceptance tests/acceptance` (new API by default; `REQCLUSTER_ACCEPTANCE_API=legacy` for the old one, baseline 53/104; slice categories 1,2,3,6,7,8 legacy 34/62).

## Status (2026-11-19, evening)
Done and committed: backend S0-S18 (+S19 folded into S14-S16), frontend S20-S23 (axios and utils/api.js removed), eval cutover S24 with fresh fast-suite baselines (all pass), review fixes, enriched/hybrid modes, no-PCA UMAP (05 H).
Acceptance 104/104 on the new API (legacy 53/104). Gate: `uv run python docs/redesign/findings/exp/newpipe_gate.py var/cache/gate` (ARI uav2k .976, uav10k .966, multidomain .671, adversarial .414).
In progress: S30 backend feature parity (extra upload columns, embedding comparison/ablation, enrichment quality, dependency rationale document, coherence), then S30 frontend wiring.
S25 scripts and S26 legacy delete done. Next: infra S27. Not yet run in a browser.
Do not `git push` until the user says so.
