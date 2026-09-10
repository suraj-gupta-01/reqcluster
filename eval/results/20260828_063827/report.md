# ReqCluster Evaluation Report

- **Timestamp:** `2026-08-28T06:37:25.393182+00:00`
- **Commit:** `78b1629`
- **Model Mode:** `real`
- **LLM Provider:** `mock`
- **Overall Suite Verdict:** **✅ PASS**

---

## Dataset: `aerospace`

| Metric | Baseline | Actual | Delta | Verdict | Notes |
|---|---:|---:|---:|:---:|---|
| `accuracy_pct` | 58.62 | 58.62 | 0 | **PASS** | Within tolerances |
| `reference_recall` | 1 | 1 | 0 | **PASS** | Within tolerances |
| `merge_precision` | 1 | 1 | 0 | **PASS** | Within tolerances |
| `parse_success_rate` | 1 | 1 | 0 | **PASS** | Within tolerances |
| `n` | 64 | 64 | 0 | **INFO** | Informational (not gated) |
| `n_clusters` | 5 | 5 | 0 | **INFO** | Informational (not gated) |
| `noise` | 6 | 6 | 0 | **INFO** | Informational (not gated) |
| `noise_pct` | 9.38 | 9.38 | 0 | **INFO** | Informational (not gated) |
| `coverage_pct` | 90.62 | 90.62 | 0 | **INFO** | Informational (not gated) |
| `silhouette` | 0.4395 | 0.4395 | 0 | **INFO** | Informational (not gated) |
| `purity_pct` | 58.62 | 58.62 | 0 | **INFO** | Informational (not gated) |
| `ari` | 0.503 | 0.503 | 0 | **INFO** | Informational (not gated) |
| `nmi` | 0.7102 | 0.7102 | 0 | **INFO** | Informational (not gated) |
| `v_measure` | 0.7102 | 0.7102 | 0 | **INFO** | Informational (not gated) |
| `homogeneity` | 0.62 | 0.62 | 0 | **INFO** | Informational (not gated) |
| `completeness` | 0.831 | 0.831 | 0 | **INFO** | Informational (not gated) |
| `n_true_groups` | 8 | 8 | 0 | **INFO** | Informational (not gated) |
| `reference_precision` | 0.5333 | 0.5333 | 0 | **INFO** | Informational (not gated) |
| `reference_f1` | 0.6957 | 0.6957 | 0 | **INFO** | Informational (not gated) |
| `n_expected_reference_edges` | 8 | 8 | 0 | **INFO** | Informational (not gated) |
| `n_inferred_reference_edges` | 15 | 15 | 0 | **INFO** | Informational (not gated) |
| `n_inferred_total_edges` | 61 | 61 | 0 | **INFO** | Informational (not gated) |
| `total_enrichment_rows` | 64 | 64 | 0 | **INFO** | Informational (not gated) |
| `succeeded_enrichment_rows` | 64 | 64 | 0 | **INFO** | Informational (not gated) |
| `mean_hallucination_risk_score` | 0.0305 | 0.0305 | 0 | **INFO** | Informational (not gated) |
| `mean_adjusted_confidence_score` | 0.8241 | 0.8241 | 0 | **INFO** | Informational (not gated) |
| `mean_lexical_overlap` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `mean_domain_term_coverage` | 0.936 | 0.936 | 0 | **INFO** | Informational (not gated) |
| `warning_rate` | 0.2031 | 0.2031 | 0 | **INFO** | Informational (not gated) |
| `n_merge_suggestions` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `correct_merges` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `n_eligible_merge_pairs` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `merge_recall` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `n_split_suggestions` | 2 | 2 | 0 | **INFO** | Informational (not gated) |
| `correct_splits` | 2 | 2 | 0 | **INFO** | Informational (not gated) |
| `n_impure_clusters` | 5 | 5 | 0 | **INFO** | Informational (not gated) |
| `split_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `split_recall` | 0.4 | 0.4 | 0 | **INFO** | Informational (not gated) |
| `suggestion_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |

## Dataset: `uav_fms_2k`

| Metric | Baseline | Actual | Delta | Verdict | Notes |
|---|---:|---:|---:|:---:|---|
| `accuracy_pct` | 97.65 | 97.65 | 0 | **PASS** | Within tolerances |
| `noise_pct` | 0 | 0 | 0 | **PASS** | Within tolerances |
| `ari` | 0.9592 | 0.9592 | 0 | **PASS** | Within tolerances |
| `v_measure` | 0.9761 | 0.9761 | 0 | **PASS** | Within tolerances |
| `parse_success_rate` | 1 | 1 | 0 | **PASS** | Within tolerances |
| `n` | 1998 | 1998 | 0 | **INFO** | Informational (not gated) |
| `n_clusters` | 16 | 16 | 0 | **INFO** | Informational (not gated) |
| `noise` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `coverage_pct` | 100 | 100 | 0 | **INFO** | Informational (not gated) |
| `silhouette` | 0.8325 | 0.8325 | 0 | **INFO** | Informational (not gated) |
| `purity_pct` | 97.65 | 97.65 | 0 | **INFO** | Informational (not gated) |
| `nmi` | 0.9761 | 0.9761 | 0 | **INFO** | Informational (not gated) |
| `homogeneity` | 0.9718 | 0.9718 | 0 | **INFO** | Informational (not gated) |
| `completeness` | 0.9804 | 0.9804 | 0 | **INFO** | Informational (not gated) |
| `n_true_groups` | 17 | 17 | 0 | **INFO** | Informational (not gated) |
| `reference_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `reference_recall` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `reference_f1` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `n_expected_reference_edges` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `n_inferred_reference_edges` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `n_inferred_total_edges` | 4000 | 4000 | 0 | **INFO** | Informational (not gated) |
| `total_enrichment_rows` | 1998 | 1998 | 0 | **INFO** | Informational (not gated) |
| `succeeded_enrichment_rows` | 1998 | 1998 | 0 | **INFO** | Informational (not gated) |
| `mean_hallucination_risk_score` | 0.0475 | 0.0475 | 0 | **INFO** | Informational (not gated) |
| `mean_adjusted_confidence_score` | 0.8096 | 0.8096 | 0 | **INFO** | Informational (not gated) |
| `mean_lexical_overlap` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `mean_domain_term_coverage` | 0.9902 | 0.9902 | 0 | **INFO** | Informational (not gated) |
| `warning_rate` | 0.3168 | 0.3168 | 0 | **INFO** | Informational (not gated) |
| `n_merge_suggestions` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `correct_merges` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `n_eligible_merge_pairs` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `merge_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `merge_recall` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `n_split_suggestions` | 5 | 5 | 0 | **INFO** | Informational (not gated) |
| `correct_splits` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `n_impure_clusters` | 2 | 2 | 0 | **INFO** | Informational (not gated) |
| `split_precision` | 0.2 | 0.2 | 0 | **INFO** | Informational (not gated) |
| `split_recall` | 0.5 | 0.5 | 0 | **INFO** | Informational (not gated) |
| `suggestion_precision` | 0.2 | 0.2 | 0 | **INFO** | Informational (not gated) |

## Dataset: `promise`

| Metric | Baseline | Actual | Delta | Verdict | Notes |
|---|---:|---:|---:|:---:|---|
| `parse_success_rate` | 1 | 1 | 0 | **PASS** | Within tolerances |
| `n` | 968 | 968 | 0 | **INFO** | Informational (not gated) |
| `n_clusters` | 19 | 19 | 0 | **INFO** | Informational (not gated) |
| `noise` | 115 | 115 | 0 | **INFO** | Informational (not gated) |
| `noise_pct` | 11.88 | 11.88 | 0 | **INFO** | Informational (not gated) |
| `coverage_pct` | 88.12 | 88.12 | 0 | **INFO** | Informational (not gated) |
| `silhouette` | 0.4708 | 0.4708 | 0 | **INFO** | Informational (not gated) |
| `accuracy_pct` | 60.49 | 60.49 | 0 | **INFO** | Informational (not gated) |
| `purity_pct` | 60.49 | 60.49 | 0 | **INFO** | Informational (not gated) |
| `ari` | 0.0745 | 0.0745 | 0 | **INFO** | Informational (not gated) |
| `nmi` | 0.3053 | 0.3053 | 0 | **INFO** | Informational (not gated) |
| `v_measure` | 0.3053 | 0.3053 | 0 | **INFO** | Informational (not gated) |
| `homogeneity` | 0.3836 | 0.3836 | 0 | **INFO** | Informational (not gated) |
| `completeness` | 0.2535 | 0.2535 | 0 | **INFO** | Informational (not gated) |
| `n_true_groups` | 12 | 12 | 0 | **INFO** | Informational (not gated) |
| `reference_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `reference_recall` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `reference_f1` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `n_expected_reference_edges` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `n_inferred_reference_edges` | 0 | 0 | 0 | **INFO** | Informational (not gated) |
| `n_inferred_total_edges` | 2143 | 2143 | 0 | **INFO** | Informational (not gated) |
| `total_enrichment_rows` | 968 | 968 | 0 | **INFO** | Informational (not gated) |
| `succeeded_enrichment_rows` | 968 | 968 | 0 | **INFO** | Informational (not gated) |
| `mean_hallucination_risk_score` | 0.0473 | 0.0473 | 0 | **INFO** | Informational (not gated) |
| `mean_adjusted_confidence_score` | 0.8098 | 0.8098 | 0 | **INFO** | Informational (not gated) |
| `mean_lexical_overlap` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `mean_domain_term_coverage` | 0.973 | 0.973 | 0 | **INFO** | Informational (not gated) |
| `warning_rate` | 0.3027 | 0.3027 | 0 | **INFO** | Informational (not gated) |
| `n_merge_suggestions` | 2 | 2 | 0 | **INFO** | Informational (not gated) |
| `correct_merges` | 2 | 2 | 0 | **INFO** | Informational (not gated) |
| `n_eligible_merge_pairs` | 69 | 69 | 0 | **INFO** | Informational (not gated) |
| `merge_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `merge_recall` | 0.029 | 0.029 | 0 | **INFO** | Informational (not gated) |
| `n_split_suggestions` | 5 | 5 | 0 | **INFO** | Informational (not gated) |
| `correct_splits` | 5 | 5 | 0 | **INFO** | Informational (not gated) |
| `n_impure_clusters` | 18 | 18 | 0 | **INFO** | Informational (not gated) |
| `split_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |
| `split_recall` | 0.2778 | 0.2778 | 0 | **INFO** | Informational (not gated) |
| `suggestion_precision` | 1 | 1 | 0 | **INFO** | Informational (not gated) |

---


=============================================================================================================
                          Metric Drift History
=============================================================================================================
 Timestamp            Dataset         Commit   Model   Accuracy      ARI    Noise   Ref Recall   Parse Rate
-------------------------------------------------------------------------------------------------------------
 2026-08-28 06:22:03  uav_fms_2k      f7cd965  real      97.65%   0.9592    0.00%       100.0%       100.0%
 2026-08-28 06:22:15  promise         f7cd965  real      60.49%   0.0745   11.88%       100.0%       100.0%
 2026-08-28 06:37:52  aerospace       78b1629  real      58.62%   0.5030    9.38%       100.0%       100.0%
 2026-08-28 06:38:14  uav_fms_2k      78b1629  real      97.65%   0.9592    0.00%       100.0%       100.0%
 2026-08-28 06:38:27  promise         78b1629  real      60.49%   0.0745   11.88%       100.0%       100.0%
=============================================================================================================

