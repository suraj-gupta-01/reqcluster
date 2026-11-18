# ReqCluster Evaluation Harness — Design Spec

**Date:** 2026-08-20
**Status:** Approved (design review)
**Owner:** RDF

## 1. Problem

The hackathon is over. ReqCluster is moving toward industry use, where every
change to the pipeline must not silently degrade result quality. Today quality is
tracked ad hoc: unit tests assert plumbing, `metrics.md` records hand-measured
numbers, and validation metrics (`/api/metrics`) exist but are only computed
interactively in the UI. There is no systematic, reproducible way to:

- prove that a code change kept clustering/dependency/enrichment/refinement
  quality stable,
- catch slow quality decay over time,
- evaluate a new dataset (including the official ones coming soon) against a
  known baseline.

## 2. Goal

A **regression guard + benchmark suite**: a hermetic, deterministic, dataset-agnostic
harness that drives the real FastAPI + database stack end-to-end, scores results
across four quality dimensions, compares them against committed golden baselines,
and fails CI on any regression.

## 3. Non-goals (v1)

- Live-network `--live` server mode (design retained for later).
- DB-persisted results / dashboard.
- Official-dataset specifics (the registry makes them plug-in with no code change).
- Latency / throughput / cost benchmarks.

## 4. Architecture

### 4.1 Execution model (Approach A — in-process E2E)

The harness boots the **real** `backend.main:app` inside `fastapi.testclient.TestClient`
against a fresh temp SQLite database. This is the same in-process HTTP-level pattern
already proven by `tests/test_async_clustering.py`: requests go through the real
routes, services, DB ORM, background-task machinery, and `/api/progress` polling —
no live port, no external services, fully hermetic.

### 4.2 Hermetic environment per run

Before importing the backend, the CLI sets:

| Setting | Value |
|---|---|
| `DATABASE_URL` | `sqlite:///<tempdir>/eval.db` |
| `REDIS_URL` | unset (file cache) |
| `REQCLUSTER_LLM_PROVIDER` | `mock` by default; `local`/`openai_compatible` via `--provider` |
| embedding cache dir | monkeypatched `core.embeddings.CACHE_DIR` → `<tempdir>/embeddings` |

Nothing in the working tree is touched except `eval/results/`, `eval/baselines/`,
`eval/drift.json`.

### 4.3 Model modes

| Mode | Model | Quality gates |
|---|---|---|
| `--model real` (default) | real `all-MiniLM-L6-v2` SBERT | yes — the real baselines |
| `--model mock` | dummy deterministic encoder (conftest-style) | no — pipeline smoke only |

CI caches the HuggingFace model dir so the ~80 MB download happens once.

## 5. Repository layout

```
eval/
  __init__.py
  __main__.py            # → cli.main()
  cli.py                 # argparse: run / update-baselines / report / drift
  config.py              # loads eval/datasets.yaml, tolerances, paths
  datasets.yaml          # dataset registry + per-dataset suite tier + tolerances
  runner.py              # boot app, TestClient, env/tempdir setup, model mode
  pipeline.py            # orchestrates the HTTP call sequence per dataset
  metrics.py             # clustering metrics passthrough (from /api/metrics)
  deps_metrics.py        # reference/data-edge precision & recall
  enrich_metrics.py      # parse-success + quality-report aggregates
  refine_metrics.py      # label-based suggestion precision & recall
  baselines.py           # golden baseline load/compare (delta tolerances)
  drift.py               # append/read eval/drift.json
  report.py              # Markdown report builder
tests/
  test_eval_harness.py   # offline unit tests for the harness itself
.github/workflows/eval.yml
```

## 6. Dataset registry (`eval/datasets.yaml`)

Each entry maps a name to data and declares its ground truth and gating:

```yaml
datasets:
  aerospace:
    path: data/aerospace_requirements.csv
    suite: fast                 # fast | stress
    ground_truth: module        # column carrying true labels
    edges: data/aerospace_requirements.edges.json   # optional dep ground truth
    tolerances:
      accuracy_pct:       {floor: 90.0, delta: -0.5}          # higher-is-better
      noise_pct:          {floor: 0.0, delta: +1.0, lower_is_better: true}
      reference_recall:   {floor: 0.8, delta: -0.05}
```

`suite: fast` runs in CI; `suite: stress` runs on demand / pre-release. Official
datasets are added by appending a block — no code change.

## 7. Metrics and ground truth

All scores come from real API responses, aggregated by `eval/metrics.py` et al.

### 7.1 Clustering accuracy

Source: `GET /api/metrics?session_id=` after clustering. Extrinsic block requires
the upload to carry a ground-truth column (the `module` column survives
preprocessing, see `core/preprocessing.py`).

Captured: `accuracy_pct` (purity), `ari`, `nmi`, `v_measure`, `noise_pct`,
`n_clusters`, `coverage_pct`, `silhouette`.

### 7.2 Dependency tree quality

Source: `GET /api/dependencies?session_id=` after
`POST /api/dependencies/generate`.

Ground truth = set of expected edges, two sources:

1. **Reference edges** — derived from the dataset text itself: for every
   requirement whose text mentions another requirement's id (`REQ-###`), an
   expected `reference` edge `mentioned → mentioning` (edge direction matches the
   tree's convention: source is prerequisite of target).
2. **Data/producer-consumer edges** — optional `<dataset>.edges.json` manifest:
   `[{source, target, relation}]` by req id, for datasets that declare them.

Inferred edges are read from `response.rationale.dependencies`
(`source_req_id`, `target_req_id`, `relation`).

Metrics: `reference_precision`, `reference_recall`, `reference_f1`,
`data_precision`, `data_recall`, `data_f1` (data metrics only when a manifest
exists), plus `n_edges` and `n_reference_edges` for context.

### 7.3 LLM enrichment quality

Source: `POST /api/enrich` + `GET /api/enrich/results?session_id=`.

The harness uses the **deterministic mock provider in CI** so results are
reproducible. Metrics from `GET /api/enrich/results` (each row carries a
`quality_report` produced by `llm_services/quality.py`):

| Metric | Derivation |
|---|---|
| `parse_success_rate` | succeeded / total |
| `mean_hallucination_risk_score` | mean of per-row `hallucination_risk_score` |
| `mean_adjusted_confidence_score` | mean of per-row `adjusted_confidence_score` |
| `mean_lexical_overlap` | mean of per-row `lexical_overlap` |
| `warning_rate` | rows with ≥1 warning / succeeded |
| `mean_domain_term_coverage` | mean of per-row `domain_term_coverage` |

### 7.4 Refinement suggestions

Source: `POST /api/suggestions/generate` + `GET /api/suggestions?session_id=`.

Judgment uses the dataset's ground-truth labels — no crafted fixtures:

- **A merge suggestion (`cluster_a_id`, `cluster_b_id`) is correct** iff the two
  clusters share the same dominant true label.
- **A split suggestion (`cluster_id`) is correct** iff that cluster is
  label-impure (its majority label covers < 100% of members).

Per-requirement labels come from `GET /api/requirements?session_id=` (each row has
`module` + `cluster_id`).

Metrics: `merge_precision`, `merge_recall`, `split_precision`, `split_recall`,
`suggestion_precision` (combined). Recall denominators: merge — the number of
cluster pairs sharing a dominant label; split — the number of label-impure
clusters.

## 8. Orchestration per dataset

`eval/pipeline.py` drives one dataset through the exact user journey:

```
upload CSV  →  poll /api/progress until done  →  /api/metrics
            →  /api/dependencies/generate (+ GET)  →  /api/enrich  →  /api/enrich/results
            →  /api/suggestions/generate  →  /api/suggestions  →  /api/requirements
```

Clustering runs as a background task; the harness polls `GET /api/progress/{id}`
(1 s TTL) until `step == "done"`. Each stage failure is captured per dataset and
reported as FAIL — never a silent skip.

## 9. Baselines, gating, and drift

- **Golden baselines:** `eval/baselines/<dataset>.json`, one per dataset, per
  provider mode (`mock`/`local`). Written only by the explicit
  `python -m eval update-baselines` command.
- **Gating:** each metric can declare an absolute `floor` and/or a signed `delta`
  tolerance in `datasets.yaml`. Direction defaults to higher-is-better; a metric
  with `lower_is_better: true` (e.g. noise, warning rate) inverts both checks. A
  metric PASSES iff the run value is on the good side of both. Missing baseline
  → FAIL with "no baseline" (forces an explicit re-baseline before a metric can
  gate).
- **Drift:** `eval/drift.json`, append-only history
  `[{timestamp, commit, dataset, provider, metrics}]`. `eval drift` prints the
  last N runs per dataset to surface slow decay.

## 10. Reports

Per run: `eval/results/<timestamp>/report.md` — a table per dataset
(*baseline → actual → delta → PASS/FAIL*), the drift summary, and an overall
verdict. Exit code 0 if all gated metrics pass, 1 otherwise (CI fails cleanly).

## 11. CLI

```
python -m eval run [--model real|mock] [--provider mock|local|openai_compatible]
                   [--suite fast|stress|all] [--dataset NAME]
python -m eval update-baselines [--model real] [--provider mock]
python -m eval report [--latest]
python -m eval drift [--n 10]
```

## 12. CI integration (`.github/workflows/eval.yml`)

New job on push/PR: pip install deps → cache HuggingFace + numba → 
`python -m eval run --model real` on the fast suite → fail on any violation.
Runs in parallel with the existing backend/frontend jobs.

## 13. Error handling

- Per-dataset try/except in `pipeline.py`; exceptions become FAIL rows with the
  error message in the report.
- Hermetic temp dirs are cleaned up even on failure (`atexit`/context manager).
- Determinism note: UMAP is seeded below N≈4000 (small datasets), so fast-suite
  runs are reproducible; the harness pins the same seeds and provider.

## 14. Harness tests

`tests/test_eval_harness.py` (offline, no model, no server):

- baseline compare + delta/floor math,
- drift append/read,
- reference-edge precision/recall against a tiny hand-built fixture,
- label-based merge/split judgment,
- tolerance config parsing.

## 15. Future (explicitly deferred)

- `--live` server mode over HTTP (`httpx`) for deployed-instance validation.
- DB-persisted results + dashboard.
- Latency/cost benchmarks.
- Official datasets (registry-ready).