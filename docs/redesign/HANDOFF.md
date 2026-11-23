# Handoff (2026-11-24)

Read this first when resuming. The branch is `overhaul`; it is pushed to `origin/overhaul`. Then read `docs/redesign/README.md` (status) and `docs/system-test-and-debug.md` (the latest external review with experiments).

## Rules the user set
- Root-cause fixes only. Never a workaround. Every bug fix ships with a test that failed before the fix.
- Commits have **no** `Co-Authored-By` trailer. Push only to `overhaul`, and only when the user asks. Open no PR unless asked.
- Keep UMAP. Do not swap the reducer.
- Never touch the user's own `pg` Postgres container.
- Agent routing per `CLAUDE.md`, **actually used**: scout (haiku), builder (sonnet), architect (opus). Builder specs carry a token budget (pipe output through `tail`/grep, no full-file reads, no exploration).
- If a question to the user goes unanswered for about a minute, take the recommended option and continue.

## State (2026-11-24)
Local commits on `overhaul`, **not pushed**:
- `805a54c` A later human edit supersedes older constraints it contradicts. Moving two items out of one cluster stored cannot(a,b) and then must(a,b), and recluster failed. Constraints are active only while the placement agrees with them. Also adds `tests/api/test_workflows.py`: chained workflows cross-checking requirements, clusters, detail, graph, metrics, session and CSV after every step, including the 0- and 1-cluster states.
- `47f081e` Eval gates `accuracy_noise_inclusive_pct` on every labelled fast dataset (finding #3). `data/sample_requirements.csv` had 2 rows with unquoted commas; a test now checks every data CSV.
- `9e59610` Ward membership is a centroid silhouette, not a 1.0 placeholder.
- `fcc1fed` For 20-2000 rows (hdbscan): consensus of 5 UMAP seeds plus radius rescue. On 8 labelled sets x 10 seeds, mean ARI MiniLM .311 -> .374, gte .348 -> .383, noise ~12% -> ~4%. The eval fast suite passes; noise-inclusive accuracy bit_control 43 -> 49, aerospace 58 -> 70, multidomain 74 -> 81. Experiment: `docs/redesign/findings/exp/small_accuracy.py` (caches in `var/cache/gate/`).
- Tests: unit+api 375, acceptance+hardening 175, eval 15, all pass; ruff clean.

Why the ARI ceiling differs per set: `promise` labels are requirement *types* and `dependency_gold` labels are dependency *roles*. Topic clustering cannot match those, so low ARI there is expected. `bit_control`'s 8 groups overlap in theme.

## Open work (priority order)
1. **Accuracy, next levers** (test on all labelled sets and several seeds, never tuned on one):
   - ~~Target cluster count `k`~~ tested 2026-11-24, **rejected**: Ward on x10 cut at the true k gives mean ARI gte .383 -> .413, MiniLM .374 -> .382, but wins only 10/16 model x set pairs (stress_600 .568 -> .459), and that is with an oracle k. Average-linkage on co-association collapses (ARI 0 on 2 sets). Ward on full embeddings is .93 on stress_600 but .735 vs .88 on the large sets. Data: `var/cache/gate/small_accuracy_k*.jsonl`.
   - ~~gte on CPU~~ **rejected**: 7.9 texts/s vs MiniLM 77 on CPU for +.009 mean ARI. MiniLM stays the CPU default.
   - Enriched/hybrid text via the LLM (needs Ollama; slow).
   - Eval baselines are still the pre-consensus values. Re-baseline only when the user agrees.
2. **Report findings still open:** #4 (the PDF caps listings at 25; CSV is complete), #12 (active learning ranks by margin: intended), architecture items 1-9 (provenance snapshot, artifact freshness, single-process job runner, separate SQL/npz lifecycles, no schema migration).
3. **Frontend:** browser click-through of the action buttons (accept suggestion, move requirement, export download, feedback). Rendering is verified; the actions are not.

## Environment notes
- **Headroom** token-saving proxy, v0.43.1 (`uv tool`), runs as the systemd user service `headroom.service` on `127.0.0.1:8787`, with telemetry off.
  - Claude Code routes through it via `~/.claude/settings.json`, which sets `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` and `ENABLE_TOOL_SEARCH=true`.
  - Check it with `headroom doctor`.
  - Remote Control (`/rc`) is unavailable while routed.
  - To undo, remove those two env keys.
- Docker bridge networking on this host is broken (likely ufw). Build and run with `--network host`.
- The system clock was fixed with `timedatectl set-local-rtc 1`. NTP still does not sync.
- UI check script: puppeteer-core, plus the Chrome binary at `~/.cache/ms-playwright/chromium_headless_shell-1248/`. Start the app with `REQCLUSTER_HOME=<tmp> REQCLUSTER_EMBEDDING_DEVICE=cpu uv run python scripts/dev.py --no-browser`.
