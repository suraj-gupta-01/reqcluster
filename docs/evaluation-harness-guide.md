# ReqCluster Evaluation Harness & Quality Guide

This guide provides a comprehensive overview of the **ReqCluster Evaluation Harness & Regression Guard Suite** (`eval/`), detailing its execution mechanics, metric definitions, golden baseline system, and terminal output interpretation.

---

## 1. Why the Evaluation Harness Exists

ReqCluster combines machine learning (SBERT embeddings, UMAP dimensionality reduction, HDBSCAN clustering), graph inference, and LLM augmentation. 

Whenever changes are made to embeddings, prompt templates, clustering hyperparameters, or pre/post-processing logic, quality cannot silently regress. The evaluation harness converts quality from a subjective, manual inspection into an **automated, hermetic CI gate**.

### Key Architectural Principles
* **In-Process Hermetic Execution:** Uses FastAPI's `TestClient` with temporary SQLite databases (`<tmp>/eval.db`) and ephemeral embedding caches (`<tmp>/embeddings`). No external Docker containers, PostgreSQL servers, Redis instances, or GPU runners are required for CI.
* **Deterministic LLM Mode:** CI runs use the `mock` provider, ensuring reproducible numbers across platforms.
* **Committed Golden Baselines:** Reference numbers are committed in version control (`eval/baselines/`). Any code change that drops scores below defined delta/floor thresholds fails CI.
* **Append-Only Drift Tracking:** Every evaluation appends results to `eval/drift.json` to monitor long-term decay.

---

## 2. End-to-End Evaluation Flow

For each dataset defined in `eval/datasets.yaml`, the harness executes the user journey via HTTP:

```
1. POST /api/upload                (Ingest requirements CSV)
2. POST /api/cluster               (Run embeddings + UMAP + HDBSCAN)
3. Poll GET /api/progress          (Wait for pipeline completion)
4. GET  /api/metrics               (Extract clustering validation scores)
5. POST /api/dependencies/generate (Infer dependency DAG)
6. POST /api/enrich                (Run requirement expansion & quality checks)
7. POST /api/suggestions/generate  (Generate merge/split suggestions)
8. Gating & Reporting              (Compare to baselines, append drift, output report)
```

---

## 3. The 4 Metric Dimensions Explained

### 3.1 Dimension 1: Clustering Quality (`eval/metrics.py`)

| Metric | Range | Meaning & Interpretation |
|---|:---:|---|
| `accuracy_pct` (Purity) | 0% – 100% | **Cluster Purity:** Percentage of requirements whose cluster matches the majority ground-truth module of that cluster. Higher is better. |
| `ari` (Adjusted Rand Index) | -1.0 – 1.0 | Measures agreement between predicted clusters and true groups, adjusted for chance. 1.0 = perfect match, 0.0 = random clustering. |
| `nmi` (Normalized Mutual Info) | 0.0 – 1.0 | Information-theoretic similarity between cluster assignments and true labels, normalized between 0 and 1. |
| `v_measure` | 0.0 – 1.0 | Harmonic mean of **homogeneity** (each cluster contains only members of a single class) and **completeness** (all members of a class are assigned to the same cluster). |
| `silhouette` | -1.0 – 1.0 | Geometric separation of clusters in 2D space. High scores indicate dense, well-separated clusters. |
| `noise_pct` | 0% – 100% | Percentage of requirements HDBSCAN marked as noise (`cluster_id = -1`). Lower is better. |
| `coverage_pct` | 0% – 100% | Percentage of clustered non-noise points ($100 - \text{noise\_pct}$). |

---

### 3.2 Dimension 2: Dependency Tree Quality (`eval/deps_metrics.py`)

Ground-truth reference links are derived by scanning requirement texts for explicit ID mentions (`REQ-001`, `SR_12`, `SYS-101`).

| Metric | Range | Meaning & Interpretation |
|---|:---:|---|
| `reference_precision` | 0.0 – 1.0 | $\frac{|\text{Inferred References} \cap \text{Expected References}|}{|\text{Inferred References}|}$. Measures how many discovered references are genuine. |
| `reference_recall` | 0.0 – 1.0 | $\frac{|\text{Inferred References} \cap \text{Expected References}|}{|\text{Expected References}|}$. Measures how many expected requirement cross-references were successfully recovered. |
| `reference_f1` | 0.0 – 1.0 | Harmonic mean of reference precision and recall. |
| `n_inferred_total_edges` | Count | Total graph edges generated (reference + data flow + sequential). |

---

### 3.3 Dimension 3: LLM Enrichment Quality (`eval/enrich_metrics.py`)

Extracted from each requirement's structured enrichment output and verified by `llm_services/quality.py`:

| Metric | Range | Meaning & Interpretation |
|---|:---:|---|
| `parse_success_rate` | 0.0 – 1.0 | Ratio of requirements successfully expanded into valid JSON schemas without parser failures. |
| `mean_hallucination_risk_score` | 0.0 – 1.0 | Evaluates whether the LLM invented numerical constants, obligations, or false claims not in the original text. Lower is better. |
| `mean_adjusted_confidence_score` | 0.0 – 1.0 | LLM self-confidence calibrated by lexical consistency penalties. |
| `mean_lexical_overlap` | 0.0 – 1.0 | Proportion of key terms from the original requirement preserved in the expanded output. |
| `mean_domain_term_coverage` | 0.0 – 1.0 | Proportion of extracted domain vocabulary keywords explicitly used in the expansion. |
| `warning_rate` | 0.0 – 1.0 | Percentage of requirements triggering quality audit flags (e.g. length drift, token omission). Lower is better. |

---

### 3.4 Dimension 4: Refinement Suggestions (`eval/refine_metrics.py`)

Judges automated merge and split proposals against ground-truth labels without manual intervention:

| Metric | Range | Meaning & Interpretation |
|---|:---:|---|
| `merge_precision` | 0.0 – 1.0 | Proportion of merge suggestions where both clusters share the same dominant ground-truth subsystem. |
| `merge_recall` | 0.0 – 1.0 | Proportion of distinct clusters sharing the same dominant subsystem that the engine proposed merging. |
| `split_precision` | 0.0 – 1.0 | Proportion of split suggestions targeting genuinely impure clusters (majority label $< 90\%$). |
| `split_recall` | 0.0 – 1.0 | Proportion of impure clusters that were detected and proposed for splitting. |
| `suggestion_precision` | 0.0 – 1.0 | Combined precision across all merge and split proposals. |

---

## 4. Golden Baselines & Gating Rules

### Baseline Files (`eval/baselines/<dataset>_<provider>.json`)
Each baseline stores recorded metrics associated with a specific git commit:

```json
{
  "dataset": "uav_fms_2k",
  "provider": "mock",
  "created_at": "2026-08-28T06:19:03Z",
  "commit": "f7cd965",
  "metrics": {
    "accuracy_pct": 97.65,
    "ari": 0.9592,
    "noise_pct": 0.0,
    "parse_success_rate": 1.0
  }
}
```

### Gating Logic (`eval/baselines.py`)
A dataset passes CI if all gated metrics in `eval/datasets.yaml` meet their conditions:

1. **Absolute Floor (`floor`):**
   - Higher-is-better: $\text{Actual} \ge \text{Floor}$ (e.g. $\text{Accuracy} \ge 95.0\%$).
   - Lower-is-better: $\text{Actual} \le \text{Floor}$ (e.g. $\text{Noise} \le 10.0\%$).
2. **Delta Tolerance (`delta`):**
   - Signed margin against baseline: $\text{Actual} \ge \text{Baseline} + \text{Delta}$ (e.g. $\text{Delta} = -1.5\%$ allows accuracy to drop at most $1.5\%$).
3. **Missing Baseline:**
   - If a gated metric has no committed baseline, it **fails** immediately to prevent un-baselined gating.

---

## 5. Terminal Output Walkthrough & Interpretation

When executing `uv run python -m eval run --suite fast` or `uv run python -m eval report`, the terminal displays:

```text
================================================================================================
                      ReqCluster Evaluation Benchmark Report
================================================================================================
 Run Time: 2026-08-28 06:39:04 UTC  |  Commit: 78b1629  |  Model: real  |  Provider: mock
 Overall Suite Verdict: [ALL PASSED]
================================================================================================
```

### Dataset 1: `aerospace` (64 requirements, 8 true subsystems)
```text
[DATASET] aerospace (N=64, Clusters=5)
------------------------------------------------------------------------------------------------
 Metric                               Baseline       Actual      Delta  Verdict  Notes
------------------------------------------------------------------------------------------------
 accuracy_pct                            58.62        58.62          0 [PASS ] Within tolerances
 reference_recall                            1            1          0 [PASS ] Within tolerances
 merge_precision                             1            1          0 [PASS ] Within tolerances
 parse_success_rate                          1            1          0 [PASS ] Within tolerances
 n                                          64           64          0 [INFO ] Informational (not gated)
 ...
```
* **Why `accuracy_pct` is 58.62%:** `aerospace_requirements.csv` contains 64 requirements across 8 fine-grained subsystems. Unsupervised HDBSCAN aggregates them into 5 natural clusters. Purity is calibrated to 58.62% with a 55.0% floor.
* **Reference Recall is 1.0 (100%):** All 8 explicit cross-references (`REQ-002 -> REQ-001`, `REQ-011 -> REQ-009`, etc.) were recovered.
* **Merge Precision is 1.0 (100%):** All suggested cluster merges united matching subsystems.

---

### Dataset 2: `uav_fms_2k` (1,998 requirements, 17 true subsystems)
```text
[DATASET] uav_fms_2k (N=1998, Clusters=16)
------------------------------------------------------------------------------------------------
 Metric                               Baseline       Actual      Delta  Verdict  Notes
------------------------------------------------------------------------------------------------
 accuracy_pct                            97.65        97.65          0 [PASS ] Within tolerances
 noise_pct                                   0            0          0 [PASS ] Within tolerances
 ari                                    0.9592       0.9592          0 [PASS ] Within tolerances
 v_measure                              0.9761       0.9761          0 [PASS ] Within tolerances
 parse_success_rate                          1            1          0 [PASS ] Within tolerances
 ...
```
* **High Cluster Accuracy (97.65%):** SBERT + scaled UMAP cleanly isolates 16 of the 17 true UAV flight subsystems.
* **ARI = 0.9592 & V-measure = 0.9761:** Near-perfect agreement between unsupervised clusters and ground truth.
* **Noise = 0.0%:** Every requirement is assigned to a cluster.

---

### Dataset 3: `promise` (968 requirements, 12 NFR categories)
```text
[DATASET] promise (N=968, Clusters=19)
------------------------------------------------------------------------------------------------
 Metric                               Baseline       Actual      Delta  Verdict  Notes
------------------------------------------------------------------------------------------------
 parse_success_rate                          1            1          0 [PASS ] Within tolerances
 ...
```
* Real-world software engineering non-functional requirements (NFRs).
* Used as a generalizability test across diverse requirement vocabulary.

---

### Drift History Table (`uv run python -m eval drift --n 10`)

```text
=============================================================================================================
                          Metric Drift History
=============================================================================================================
 Timestamp            Dataset         Commit   Model   Accuracy      ARI    Noise   Ref Recall   Parse Rate
-------------------------------------------------------------------------------------------------------------
 2026-08-28 06:38:14  uav_fms_2k      78b1629  real      97.65%   0.9592    0.00%       100.0%       100.0%
 2026-08-28 06:38:27  promise         78b1629  real      60.49%   0.0745   11.88%       100.0%       100.0%
 2026-08-28 06:39:04  aerospace       78b1629  real      58.62%   0.5030    9.38%       100.0%       100.0%
 2026-08-28 06:39:26  uav_fms_2k      78b1629  real      97.65%   0.9592    0.00%       100.0%       100.0%
 2026-08-28 06:39:39  promise         78b1629  real      60.49%   0.0745   11.88%       100.0%       100.0%
=============================================================================================================
```

* **Purpose:** Detects subtle accuracy or recall degradation across commits before hard thresholds are violated.

---

## 6. CLI Command Summary

```bash
# 1. Run regression evaluation against golden baselines (CI Fast Suite)
uv run python -m eval run --suite fast

# 2. Run stress suite on large datasets (10k, 35k)
uv run python -m eval run --suite stress

# 3. Update golden baselines after intentional model or pipeline updates
uv run python -m eval update-baselines --suite fast

# 4. View latest benchmark comparison table in terminal
uv run python -m eval report

# 5. View historical metric drift table
uv run python -m eval drift --n 10

# 6. Run unit tests for harness components
uv run pytest tests/test_eval_harness.py -v
```
