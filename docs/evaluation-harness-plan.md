# ReqCluster Evaluation Harness — Human Plan

**Audience:** engineers and reviewers (not an AI agent).
**Status:** design approved; this is the working plan for implementation.
**Goal:** a hermetic, deterministic, dataset-agnostic **regression guard + benchmark
suite** that drives the real FastAPI + DB stack end-to-end, scores clustering /
dependency / enrichment / refinement quality, compares to committed golden
baselines, and fails CI on regression.

---

## 1. Why this exists

ReqCluster is past the hackathon. From here on, every change — a new embedding
model, a clustering-parameter tweak, a prompt change, a refactor — must not
silently degrade output quality. Today that is unproven: `metrics.md` records
hand-measured numbers, unit tests assert plumbing, and validation metrics exist
only interactively in the UI.

The harness makes quality **a tested, gated, tracked property of the codebase**:
reproducible numbers, committed baselines, CI enforcement, and a drift history to
catch slow decay.

## 2. Design decisions (and why)

| Decision | Choice | Rationale |
|---|---|---|
| Execution | **In-process E2E** (`TestClient` + temp SQLite) | Real routes/services/DB/background jobs exercised over HTTP, but hermetic, fast, deterministic, CI-safe. Same pattern already proven in `tests/test_async_clustering.py`. A future `--live` mode can add real-server HTTP later. |
| Gating | **Golden baselines + delta tolerances + absolute floors** | Catches small regressions above any floor; deltas are meaningful because small-N UMAP is seeded (reproducible). |
| LLM in CI | **Mock provider** (deterministic); real provider optional via `--provider` | Hermetic + reproducible in CI; real-LLM runs happen manually on a smaller subset. |
| Datasets | **Registry (`eval/datasets.yaml`)** | Official datasets plug in with zero code changes. |
| Ground truth | **`module` column + optional `.edges.json` + label-based judgment** | No crafted fixtures; reuse the labelled datasets for every dimension. |
| Results | **Committed baselines + append-only drift history + Markdown report** | Diffable, reviewable, no extra infra (no DB persistence / dashboard in v1). |

## 3. What gets evaluated (four dimensions)

1. **Clustering accuracy** — how well clusters match true subsystem labels.
2. **Dependency tree quality** — precision/recall of inferred dependency edges
   vs. known references.
3. **LLM enrichment quality** — parse success and hallucination/quality flags on
   expanded requirements.
4. **Refinement suggestions** — whether merge/split suggestions are correct
   against ground-truth labels.

## 4. Repository layout

```
eval/
  __init__.py              # package marker
  __main__.py              # enables `python -m eval ...`
  cli.py                   # argparse front door (run / update-baselines / report / drift)
  config.py                # loads datasets.yaml, resolves paths, applies defaults
  datasets.yaml            # THE registry + tolerances (see schema below)
  runner.py                # boot real app in TestClient; temp env; model mode
  pipeline.py              # per-dataset HTTP orchestration (upload → … → score)
  metrics.py               # clustering metrics (from /api/metrics)
  deps_metrics.py          # reference/data-edge precision & recall
  enrich_metrics.py        # parse-success + quality-report aggregates
  refine_metrics.py        # label-based merge/split judgment
  baselines.py             # golden baseline load/compare/delta logic
  drift.py                 # append/read drift history
  report.py                # Markdown report builder
tests/
  test_eval_harness.py     # offline tests for the harness itself
.github/workflows/eval.yml # CI job
```

Every module has one job and a narrow interface — keep it that way while
implementing.

## 5. Execution flow (how a run works)

### 5.1 Hermetic environment

Before importing the backend, the CLI:

1. Creates a temp dir (e.g. `<tmp>/eval-run-<ts>/`).
2. Sets `DATABASE_URL=sqlite:///<tmp>/eval.db` (SQLite on purpose: hermetic, no
   Postgres dependency; the real Postgres path is already covered by the app's own
   tests/CI).
3. Unsets `REDIS_URL` (file cache only).
4. Sets `REQCLUSTER_LLM_PROVIDER=mock` unless `--provider` overrides it.
5. Monkeypatches `core.embeddings.CACHE_DIR` → `<tmp>/embeddings` so the `.npy`
   embedding cache never leaks into the repo's `embeddings/`.
6. Imports `backend.main:app` and wraps it in `fastapi.testclient.TestClient`.

On exit the temp dir is removed even on failure.

### 5.2 Per-dataset orchestration (in `eval/pipeline.py`)

For each dataset in the selected suite, drive the exact user journey:

```
1. POST /api/upload          (multipart CSV)
2. POST /api/cluster         → 202 (background task starts)
3. GET  /api/progress/{id}   poll every 1 s until step == "done"
4. GET  /api/metrics         → clustering scores
5. POST /api/dependencies/generate   (provider=mock, sim_threshold=0.45, top_k=8)
   GET  /api/dependencies    → edges + rationale.dependencies (req-id edges)
6. POST /api/enrich          (provider=mock, embedding_mode=hybrid)
   GET  /api/enrich/results  → per-requirement quality_report rows
7. POST /api/suggestions/generate   (top_n_merges=5, top_n_splits=5)
   GET  /api/suggestions     → merge/split suggestions
   GET  /api/requirements    → per-requirement module + cluster_id (for judgment)
```

Each stage is wrapped so a failure becomes a FAIL row for that dataset, never a
silent skip.

### 5.3 Model modes

- `--model real` (default): real `all-MiniLM-L6-v2` SBERT. Quality gates apply.
  CI caches the HuggingFace cache dir so the ~80 MB model downloads once.
- `--model mock`: inject a deterministic dummy encoder (same idea as
  `tests/conftest.py`). Verifies the pipeline *completes* but produces no real
  cluster quality — used for cheap smoke checks only.

## 6. Dataset registry schema (`eval/datasets.yaml`)

```yaml
datasets:
  <name>:
    path: data/<file>.csv          # any CSV with a text column (+ optional module)
    suite: fast | stress           # fast runs in CI; stress on demand
    ground_truth: module           # column holding true group labels
    edges: data/<file>.edges.json  # OPTIONAL ground-truth data/producer-consumer edges
    tolerances:                    # each gated metric:
      <metric>:                    #   e.g. accuracy_pct, noise_pct, reference_recall
        floor: <float>             #   absolute floor (optional)
        delta: <float>             #   signed tolerance (optional)
        lower_is_better: <bool>    #   default false; true inverts both checks
```

Rules:
- A metric is gated only if it appears in `tolerances`.
- Direction defaults to **higher-is-better**; a metric with `lower_is_better:
  true` (e.g. `noise_pct`, `warning_rate`) inverts both checks.
- PASS requires being on the good side of **both** `floor` and `delta`
  (whichever are present).
- A dataset with no `tolerances` is reported but not gated (informational).

Expected fast-suite members today: `aerospace` (dep ground truth via `{ref}`
mentions), `uav_fms_2k` (clustering purity/ARI), `promise` (real NFR data).
Official datasets get appended as new blocks.

## 7. Metrics & scoring techniques

### 7.1 Clustering (`eval/metrics.py`)

Taken directly from `GET /api/metrics` (computed by `core/metrics.py`):
`accuracy_pct` (purity), `ari`, `nmi`, `v_measure`, `noise_pct`, `coverage_pct`,
`n_clusters`, `silhouette`. Extrinsic scores appear only when the upload carried
a ground-truth column.

### 7.2 Dependency tree (`eval/deps_metrics.py`)

**Ground truth construction:**

- *Reference edges:* scan each requirement's text for other requirements' ids
  (regex `\b(?:REQ|SR|FR|SYS|R)[-_ ]?\d{1,5}\b`). For each mention, add expected
  edge `mentioned → mentioning`, relation `reference`. This matches the tree's own
  direction convention (source is a prerequisite of target).
- *Data edges:* if a `<name>.edges.json` manifest exists, load
  `[{"source": "<reqid>", "target": "<reqid>", "relation": "<data|...>"}]` as
  expected data edges.

**Inferred edges:** from `GET /api/dependencies` → `rationale.dependencies`
(each has `source_req_id`, `target_req_id`, `relation`).

**Scores:**

```
precision = |inferred ∩ expected| / |inferred|
recall    = |inferred ∩ expected| / |expected|
f1        = 2·P·R / (P + R)   (0 when P+R = 0)
```

Computed separately for `reference` and `data` edges. Report also `n_edges` and
`n_expected_reference_edges` for context.

### 7.3 Enrichment (`eval/enrich_metrics.py`)

Source: `GET /api/enrich/results` — each row has `status` and a `quality_report`
from `llm_services/quality.py` (fields: `hallucination_risk_score`,
`adjusted_confidence_score`, `lexical_overlap`, `domain_term_coverage`,
`warnings`, `length_ratio`).

Aggregates:

| Metric | Formula |
|---|---|
| `parse_success_rate` | `succeeded / total` |
| `mean_hallucination_risk_score` | mean over succeeded rows |
| `mean_adjusted_confidence_score` | mean over succeeded rows |
| `mean_lexical_overlap` | mean over succeeded rows |
| `mean_domain_term_coverage` | mean over succeeded rows |
| `warning_rate` | rows with ≥ 1 warning / succeeded |

With the mock provider these are deterministic — perfect for regression gating.
Real-provider runs (`--provider local`) report the same numbers for human review
(no gating, since they are non-deterministic).

### 7.4 Refinement suggestions (`eval/refine_metrics.py`)

**Judgment using ground-truth labels** (no fixtures):

1. Build `cluster_id → list of module labels` from `GET /api/requirements`.
2. A **merge** suggestion (`cluster_a_id`, `cluster_b_id`) is *correct* iff both
   clusters share the same dominant label.
3. A **split** suggestion (`cluster_id`) is *correct* iff the cluster is
   label-impure (its majority label covers < 100 % of members).

**Scores:**

```
merge_precision = correct_merges / merge_suggestions
merge_recall    = correct_merges / eligible_pairs      # pairs sharing a dominant label
split_precision = correct_splits / split_suggestions
split_recall    = correct_splits / impure_clusters      # label-impure clusters
suggestion_precision = (correct_merges + correct_splits) / total_suggestions
```

Note: suggestions are evaluated against the *current* clustering labels; the
harness does not apply them.

## 8. Baselines, gating, drift

### 8.1 Golden baselines — `eval/baselines/<dataset>.json`

```json
{
  "dataset": "uav_fms_2k",
  "provider": "mock",
  "created_at": "2026-08-20T10:00:00Z",
  "commit": "016d9c3",
  "metrics": {
    "accuracy_pct": 98.3,
    "ari": 0.956,
    "noise_pct": 0.7,
    "...": "..."
  }
}
```

Written **only** by `python -m eval update-baselines` — deliberate, explicit, and
reviewable via git diff. One baseline file per dataset × provider mode.

### 8.2 Gating logic (`eval/baselines.py`)

For each gated metric `m` with tolerance `{floor?, delta?, lower_is_better?}`
against run value `v` and baseline `b`:

- Default direction is **higher-is-better**; `lower_is_better: true` inverts it.
- `floor` present → require `v` to be on the good side of `floor`.
- `delta` present → require `v` to be on the good side of `b + delta` (delta is
  signed: `delta: -0.5` lets a higher-is-better metric fall at most 0.5; `delta:
  +1.0` lets a lower-is-better metric like `noise_pct` rise at most 1.0).
- Missing baseline for a gated metric → FAIL with `no baseline` (forces an
  explicit re-baseline; a metric cannot silently start gating).

### 8.3 Drift history — `eval/drift.json`

```json
{
  "runs": [
    {
      "timestamp": "2026-08-20T10:00:00Z",
      "commit": "016d9c3",
      "dataset": "uav_fms_2k",
      "provider": "mock",
      "model": "real",
      "metrics": { "accuracy_pct": 98.3, "...": "..." }
    }
  ]
}
```

Append-only. `python -m eval drift --n 10` prints the last 10 runs per dataset so
a slow downward drift is visible before it trips a hard gate.

## 9. Reports — `eval/results/<timestamp>/report.md`

Per dataset:

```
## aerospace (mock, model=real)

| metric               | baseline | actual | delta     | verdict |
|----------------------|----------|--------|-----------|---------|
| accuracy_pct         | 92.0     | 92.5   | +0.5      | PASS    |
| reference_recall     | 0.85     | 0.80   | -0.05     | FAIL    |
| noise_pct            | 3.2      | 4.0    | +0.8      | PASS    |
```

Plus a drift summary and an overall verdict. Exit code 0 = all gated metrics pass,
1 = any FAIL (this is what breaks CI).

## 10. CLI surface

```
python -m eval run [--model real|mock] [--provider mock|local|openai_compatible]
                   [--suite fast|stress|all] [--dataset NAME]
python -m eval update-baselines [--model real] [--provider mock] [--dataset NAME]
python -m eval report [--latest]
python -m eval drift [--n 10]
```

## 11. CI integration — `.github/workflows/eval.yml`

New job, runs alongside the existing backend/frontend jobs on push/PR:

```yaml
steps:
  - checkout
  - setup-python 3.11
  - pip install -r backend/requirements.txt pytest pytest-cov
  - cache: ~/.cache/huggingface  (key: sbbert-all-minilm)
  - cache: numba JIT cache
  - run: python -m eval run --model real   # fast suite, mock provider
```

Fails on any delta/floor violation or missing baseline. Stress-suite runs happen
manually / pre-release.

## 12. Harness self-tests — `tests/test_eval_harness.py`

Offline, no model, no server:

- baseline compare + floor/delta math (incl. lower-is-better metrics, missing
  baseline → FAIL),
- drift append/read and ordering,
- reference-edge precision/recall on a tiny hand-built fixture (e.g. 4 reqs, 2
  expected edges, 1 spurious inferred edge),
- merge/split label-judgment on a synthetic cluster→label map,
- tolerance YAML parsing.

## 13. Error handling & determinism

- Every stage wrapped per-dataset; exceptions become FAIL rows with the message.
- Temp dirs cleaned up even on failure.
- Determinism: small-N UMAP is seeded; the harness fixes the same seeds and the
  mock provider, so fast-suite runs are byte-reproducible given the same input.
- If a stage is not applicable (e.g. no edges manifest), its metrics are omitted
  from the report rather than faked as zero.

## 14. Out of scope (v1) / future work

- `--live` real-server HTTP mode (Approach B) — validate deployed instances.
- DB-persisted results + a queryable dashboard.
- Latency / throughput / cost benchmarks.
- Official datasets — registry-ready; add a YAML block when they arrive.

## 15. Suggested implementation order

1. `eval/config.py` + `datasets.yaml` + tolerance parsing.
2. `eval/runner.py` — hermetic boot of the app in TestClient.
3. `eval/pipeline.py` — the HTTP orchestration.
4. `eval/metrics.py` + `deps_metrics.py` + `enrich_metrics.py` + `refine_metrics.py`.
5. `eval/baselines.py` + `drift.py` + `report.py`.
6. `eval/cli.py` + `__main__.py`.
7. `tests/test_eval_harness.py`.
8. Seed baselines for the fast suite; commit them.
9. `.github/workflows/eval.yml`.
10. Update `README.md` + `USER.md` with a "Quality / evaluation" section.