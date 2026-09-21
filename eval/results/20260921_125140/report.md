# ReqCluster Evaluation Report

- **Timestamp:** `2026-09-21T12:50:59.494247+00:00`
- **Commit:** `0ea108d`
- **Model Mode:** `mock`
- **LLM Provider:** `mock`
- **Overall Suite Verdict:** **❌ FAIL**

---

## Dataset: `bit_control`

| Metric | Baseline | Actual | Delta | Verdict | Notes |
|---|---:|---:|---:|:---:|---|
| `ari` | - | 0.0178 | - | **FAIL** | Missing baseline for gated metric |
| `noise_pct` | - | 42 | - | **FAIL** | Missing baseline for gated metric |
| `hungarian_accuracy_pct` | - | 32.76 | - | **FAIL** | Missing baseline for gated metric |
| `parse_success_rate` | - | 1 | - | **FAIL** | Missing baseline for gated metric |
| `n` | - | 100 | - | **INFO** | Informational (not gated) |
| `n_clusters` | - | 6 | - | **INFO** | Informational (not gated) |
| `noise` | - | 42 | - | **INFO** | Informational (not gated) |
| `coverage_pct` | - | 58 | - | **INFO** | Informational (not gated) |
| `silhouette` | - | 0.1073 | - | **INFO** | Informational (not gated) |
| `accuracy_pct` | - | 37.93 | - | **INFO** | Informational (not gated) |
| `purity_pct` | - | 37.93 | - | **INFO** | Informational (not gated) |
| `nmi` | - | 0.1829 | - | **INFO** | Informational (not gated) |
| `v_measure` | - | 0.1829 | - | **INFO** | Informational (not gated) |
| `homogeneity` | - | 0.1741 | - | **INFO** | Informational (not gated) |
| `completeness` | - | 0.1926 | - | **INFO** | Informational (not gated) |
| `n_true_groups` | - | 8 | - | **INFO** | Informational (not gated) |
| `reference_precision` | - | 1 | - | **INFO** | Informational (not gated) |
| `reference_recall` | - | 1 | - | **INFO** | Informational (not gated) |
| `reference_f1` | - | 1 | - | **INFO** | Informational (not gated) |
| `n_expected_reference_edges` | - | 0 | - | **INFO** | Informational (not gated) |
| `n_inferred_reference_edges` | - | 0 | - | **INFO** | Informational (not gated) |
| `n_inferred_total_edges` | - | 0 | - | **INFO** | Informational (not gated) |
| `total_enrichment_rows` | - | 100 | - | **INFO** | Informational (not gated) |
| `succeeded_enrichment_rows` | - | 100 | - | **INFO** | Informational (not gated) |
| `mean_hallucination_risk_score` | - | 0 | - | **INFO** | Informational (not gated) |
| `mean_adjusted_confidence_score` | - | 0.85 | - | **INFO** | Informational (not gated) |
| `mean_lexical_overlap` | - | 1 | - | **INFO** | Informational (not gated) |
| `mean_domain_term_coverage` | - | 0.9368 | - | **INFO** | Informational (not gated) |
| `warning_rate` | - | 0 | - | **INFO** | Informational (not gated) |
| `n_merge_suggestions` | - | 0 | - | **INFO** | Informational (not gated) |
| `correct_merges` | - | 0 | - | **INFO** | Informational (not gated) |
| `n_eligible_merge_pairs` | - | 6 | - | **INFO** | Informational (not gated) |
| `merge_precision` | - | 1 | - | **INFO** | Informational (not gated) |
| `merge_recall` | - | 0 | - | **INFO** | Informational (not gated) |
| `n_split_suggestions` | - | 2 | - | **INFO** | Informational (not gated) |
| `correct_splits` | - | 2 | - | **INFO** | Informational (not gated) |
| `n_impure_clusters` | - | 6 | - | **INFO** | Informational (not gated) |
| `split_precision` | - | 1 | - | **INFO** | Informational (not gated) |
| `split_recall` | - | 0.3333 | - | **INFO** | Informational (not gated) |
| `suggestion_precision` | - | 1 | - | **INFO** | Informational (not gated) |

---


=============================================================================================================
                          Metric Drift History
=============================================================================================================
 Timestamp            Dataset         Commit   Model   Accuracy      ARI    Noise   Ref Recall   Parse Rate
-------------------------------------------------------------------------------------------------------------
 2026-08-28 06:38:27  promise         78b1629  real      60.49%   0.0745   11.88%       100.0%       100.0%
 2026-08-28 06:39:04  aerospace       78b1629  real      58.62%   0.5030    9.38%       100.0%       100.0%
 2026-08-28 06:39:26  uav_fms_2k      78b1629  real      97.65%   0.9592    0.00%       100.0%       100.0%
 2026-08-28 06:39:39  promise         78b1629  real      60.49%   0.0745   11.88%       100.0%       100.0%
 2026-09-21 12:51:40  bit_control     0ea108d  mock      37.93%   0.0178   42.00%       100.0%       100.0%
=============================================================================================================

