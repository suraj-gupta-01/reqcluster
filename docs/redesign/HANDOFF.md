# Handoff (2026-11-23)

Read this first when resuming. The branch is `overhaul`; it is pushed to `origin/overhaul`. Then read `docs/redesign/README.md` (status) and `docs/system-test-and-debug.md` (the latest external review with experiments).

## Rules the user set
- Root-cause fixes only. Never a workaround. Every bug fix ships with a test that failed before the fix.
- Commits have **no** `Co-Authored-By` trailer. Push only to `overhaul`, and only when the user asks. Open no PR unless asked.
- Keep UMAP. Do not swap the reducer.
- Never touch the user's own `pg` Postgres container.
- Agent routing per `CLAUDE.md`: scout (haiku), builder (sonnet), architect (opus).

## State
- Last pushed commits: `1c75b99` (UI fixes), then `12952f0` and `167127c` from teammate Codexhack286. Those two add the system-test report and fix PDF coverage, single-cluster Hungarian accuracy and the explanation cache. They were reviewed and are correct.
- The next commit (see git log) fixes 5 more findings from that report:
  - **#5:** the run records the effective seed. A null request seed used to be stored as null.
  - **#6:** `REQCLUSTER_EMBEDDING_REVISION` (default now `main`) is passed to the model loader, so a commit hash pins the weights. It also keys the embedding cache. The old default `"1"` was a cache version only, not a real Hugging Face ref.
  - **#7:** feedback whose constraints a recluster carries moves to the new run, so it stays reviewable and rejectable.
  - **#10:** export bundles drop dependency edges when the tree is stale.
  - **#11:** a requirement moved to noise stores probability 0.0 instead of 1.0.
  - **Latent crash fixed:** the embedding-dimension lookup evaluated the deprecated method name eagerly.
- Tests: unit+api 345 pass; acceptance+hardening 175 pass; ruff is clean. CI (ubuntu, windows, postgres, frontend, eval) was green at `1c75b99`.

## Why earlier reviews missed these
The earlier tests checked one feature at a time and asserted "no 500 / no data loss". The bugs sat between features: feedback, then recluster, then review; tree, then edit, then export. They returned 200 with wrong content. **New rule:** add workflow tests that chain steps and assert the *content* is true. Also check metrics and reports on degenerate input (0 or 1 cluster, not yet clustered).

## Open work (priority order)
1. **Workflow-correctness tests:** multi-step API sequences that assert the content, not just the status codes.
2. **Accuracy on small real uploads.** The weak spot is `data/bit_control_requirements.xlsx` (100 reqs, 8 groups): ARI about 0.19 (range 0.13-0.24 over seeds), with 0-22 noise depending on the seed. Large synthetic sets are already at 0.96-0.998. Ideas to test with the eval harness, on several datasets and seeds, never tuned on one:
   - Seed-stability: run several UMAP seeds and take consensus labels.
   - `min_samples` and `min_cluster_size` defaults for N < 200 (report Experiment B).
   - Enriched/hybrid text.
   - A top-2 candidate cluster for noise rows.

   Measure with noise-inclusive metrics. ARI can't exceed 1, so "100x" is not possible; aim for higher and more stable ARI plus less spurious noise.
3. **Report findings still open:**
   - #3: eval gates should include noise-inclusive accuracy.
   - #4: the PDF caps each listing at 25 and says so; CSV is the complete export.
   - #12: active-learning ranks by centroid margin. That is documented and intended.
   - Ward's membership values are placeholders of 1.0.
   - Architecture items 1-9: provenance snapshot, artifact freshness, single-process job runner, separate SQL and npz lifecycles, no automatic schema migration.
4. **Frontend:** a browser click-through of the actions (accept a suggestion, move a requirement, export downloads, feedback). Page rendering is verified; the action buttons are not.

## Environment notes
- **Headroom** token-saving proxy, v0.43.1 (`uv tool`), runs as the systemd user service `headroom.service` on `127.0.0.1:8787`, with telemetry off.
  - Claude Code routes through it via `~/.claude/settings.json`, which sets `ANTHROPIC_BASE_URL=http://127.0.0.1:8787` and `ENABLE_TOOL_SEARCH=true`.
  - Check it with `headroom doctor`.
  - Remote Control (`/rc`) is unavailable while routed.
  - To undo, remove those two env keys.
- Docker bridge networking on this host is broken (likely ufw). Build and run with `--network host`.
- The system clock was fixed with `timedatectl set-local-rtc 1`. NTP still does not sync.
- UI check script: puppeteer-core, plus the Chrome binary at `~/.cache/ms-playwright/chromium_headless_shell-1248/`. Start the app with `REQCLUSTER_HOME=<tmp> REQCLUSTER_EMBEDDING_DEVICE=cpu uv run python scripts/dev.py --no-browser`.
