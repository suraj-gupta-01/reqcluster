# Graph Report - backend  (2026-06-04)

## Corpus Check
- Corpus is ~26,895 words - fits in a single context window. You may not need a graph.

## Summary
- 519 nodes · 994 edges · 32 communities (23 shown, 9 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 148 edges (avg confidence: 0.73)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_LLM Provider & Enrichment|LLM Provider & Enrichment]]
- [[_COMMUNITY_Domain Embeddings & Ablation|Domain Embeddings & Ablation]]
- [[_COMMUNITY_Persistence & Phase-5 Services|Persistence & Phase-5 Services]]
- [[_COMMUNITY_API Schemas & Routes|API Schemas & Routes]]
- [[_COMMUNITY_Refinement LLM Provider|Refinement LLM Provider]]
- [[_COMMUNITY_Enrichment Persistence|Enrichment Persistence]]
- [[_COMMUNITY_Embeddings & Representatives|Embeddings & Representatives]]
- [[_COMMUNITY_MBSE Export|MBSE Export]]
- [[_COMMUNITY_Merge Suggestion|Merge Suggestion]]
- [[_COMMUNITY_Refinement Service|Refinement Service]]
- [[_COMMUNITY_Split Suggestion|Split Suggestion]]
- [[_COMMUNITY_Dependency Tree (DP5)|Dependency Tree (DP5)]]
- [[_COMMUNITY_API Routes|API Routes]]
- [[_COMMUNITY_Feedback Service|Feedback Service]]
- [[_COMMUNITY_Feedback Analyst|Feedback Analyst]]
- [[_COMMUNITY_Constrained Clustering|Constrained Clustering]]
- [[_COMMUNITY_Preprocessing|Preprocessing]]
- [[_COMMUNITY_Feedback Endpoints|Feedback Endpoints]]
- [[_COMMUNITY_Feedback Endpoints|Feedback Endpoints]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]

## God Nodes (most connected - your core abstractions)
1. `normalize_plain_text()` - 25 edges
2. `run_and_persist_enrichment()` - 18 edges
3. `enrich_requirements()` - 15 edges
4. `generate_and_persist_suggestions()` - 15 edges
5. `ProviderResponseError` - 14 edges
6. `EnrichmentServiceError` - 14 edges
7. `run_pipeline()` - 13 edges
8. `run_embedding_ablation()` - 12 edges
9. `build_dependency_tree()` - 12 edges
10. `evaluate_expansion_quality()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `get_enrichment_status_endpoint()` --calls--> `get_enrichment_status()`  [INFERRED]
  api/routes.py → services/enrichment_service.py
- `get_enrichment_results_endpoint()` --calls--> `get_enrichment_results()`  [INFERRED]
  api/routes.py → services/enrichment_service.py
- `upload_requirements()` --calls--> `preprocess_requirements()`  [INFERRED]
  api/routes.py → core/preprocessing.py
- `upload_requirements()` --calls--> `UploadResponse`  [INFERRED]
  api/routes.py → models/schemas.py
- `enrich_requirements_endpoint()` --calls--> `run_and_persist_enrichment()`  [INFERRED]
  api/routes.py → services/enrichment_service.py

## Communities (32 total, 9 thin omitted)

### Community 0 - "LLM Provider & Enrichment"
Cohesion: 0.06
Nodes (68): _bounded_float(), _bounded_int(), _cache_record(), _dedupe(), enrich_requirements(), enrich_then_prepare_pipeline_inputs(), _error(), extract_enriched_texts() (+60 more)

### Community 1 - "Domain Embeddings & Ablation"
Cohesion: 0.06
Nodes (56): _empty_variant_report(), _missing_enriched_count(), _noise_summary(), Run a read-only base-vs-domain embedding ablation report., run_embedding_ablation(), _run_variant_pipeline(), _safe_float(), _silhouette_or_none() (+48 more)

### Community 2 - "Persistence & Phase-5 Services"
Cohesion: 0.07
Nodes (40): cluster_requirements_endpoint(), Run the clustering pipeline on an uploaded session., Upload and preprocess a CSV or XLSX requirements file., upload_requirements(), lifespan(), Base, clustering_quality(), Active-learning utilities: uncertainty sampling and clustering quality.  Selects (+32 more)

### Community 3 - "API Schemas & Routes"
Cohesion: 0.11
Nodes (36): get_cluster_detail(), Get cluster details with requirements., BaseModel, ApplySuggestionRequest, ApplySuggestionResponse, AuditLogEntry, ClusterDetail, ClusterOut (+28 more)

### Community 4 - "Refinement LLM Provider"
Cohesion: 0.07
Nodes (17): RequirementExpansionProvider, ClusterRefinementProvider, ClusterSummary, CoherenceResult, get_refinement_provider(), LLMClusterRefinementProvider, MockClusterRefinementProvider, ClusterLLM coherence scoring and rationale generation.  This module provides L (+9 more)

### Community 5 - "Enrichment Persistence"
Cohesion: 0.15
Nodes (33): EnrichmentBatchResult, EnrichedRequirement, Naive UTC timestamp.      Replaces the deprecated ``datetime.utcnow`` (removed, utcnow(), build_enriched_texts_for_pipeline(), _complete_enrichment_group(), _dedupe(), EnrichmentAlignmentError (+25 more)

### Community 6 - "Embeddings & Representatives"
Cohesion: 0.1
Nodes (20): get_constraints_endpoint(), Get active constraint pairs and detect conflicts in the constraint network., _cache_key(), _cache_path(), generate_embeddings(), get_model(), Generate SBERT embeddings for a list of texts.     Returns numpy array of shape, SentenceTransformer (+12 more)

### Community 7 - "MBSE Export"
Cohesion: 0.12
Nodes (18): build_jama_bundle(), export_jama(), Jama Connect export connector.  Builds an importable Jama item bundle (the shape, _el(), export_reqif(), _q(), ReqIF 1.2 (OMG) exporter.  Produces a schema-shaped ReqIF XML document with a DA, export_sysml_xmi() (+10 more)

### Community 8 - "Merge Suggestion"
Cohesion: 0.13
Nodes (17): compute_cluster_centroids(), _compute_intra_cluster_coherence(), compute_pairwise_cluster_similarity(), evaluate_merge_silhouette(), MergeCandidate, MergeScore, MergeSuggestion, Cluster merge candidate detection via silhouette analysis.  This module identi (+9 more)

### Community 9 - "Refinement Service"
Cohesion: 0.15
Nodes (19): get_audit_log_endpoint(), list_suggestions_endpoint(), List refinement suggestions for a session, optionally filtered by status., Get audit log of applied refinements for a session., apply_suggestion(), generate_and_persist_suggestions(), get_audit_log(), get_suggestions() (+11 more)

### Community 10 - "Split Suggestion"
Cohesion: 0.14
Nodes (15): BimodalityResult, compute_cluster_spread(), evaluate_split_silhouette(), Cluster split candidate detection via bimodality and silhouette analysis.  Thi, Test whether a cluster has a bimodal distribution using GMM.      Fits 1-compo, Compute silhouette score delta if a cluster were split.      A positive delta, Result of a bimodality test on a single cluster., Full split suggestion pipeline.      1. Compute spread for each cluster. (+7 more)

### Community 11 - "Dependency Tree (DP5)"
Cohesion: 0.18
Nodes (15): _add_edge(), _assign_levels(), _break_cycles(), build_dependency_tree(), DependencyEdge, DependencyNode, _first_cue(), _normalize_id() (+7 more)

### Community 12 - "API Routes"
Cohesion: 0.13
Nodes (12): apply_suggestion_endpoint(), constrained_recluster_endpoint(), generate_dependencies_endpoint(), get_enrichment_results_endpoint(), get_enrichment_status_endpoint(), get_graph(), get_progress(), Get pipeline progress for a session. (+4 more)

### Community 13 - "Feedback Service"
Cohesion: 0.21
Nodes (11): export_feedback_endpoint(), get_feedback_queue_endpoint(), Retrieve the human feedback review queue for a session., Export the human feedback queue as CSV or JSON., export_feedback_csv(), export_feedback_json(), get_feedback_queue(), Phase 4 human feedback service layer.  Handles feedback corrections submission (+3 more)

### Community 14 - "Feedback Analyst"
Cohesion: 0.22
Nodes (5): FeedbackAnalyst, Feedback comment analyst and narrative updater.  Evaluates user correction ann, Analyzes human-in-the-loop annotations and comments., Compute a confidence score in [0.5, 1.0] based on comment overlap.          If, Produce an updated summary paragraph highlighting user refinement.

### Community 15 - "Constrained Clustering"
Cohesion: 0.48
Nodes (6): apply_constraints(), _cluster_centroids(), _find(), Constraint enforcement for clustering (Phase 5 active learning).  HDBSCAN has no, Enforce must-link / cannot-link constraints on a label assignment.      Args:, _union()

### Community 16 - "Preprocessing"
Cohesion: 0.33
Nodes (6): clean_text(), normalize_column_names(), preprocess_requirements(), Normalize column names to expected schema., Clean a single requirement text., Load and clean requirements from CSV or XLSX.     Returns cleaned DataFrame and

### Community 17 - "Feedback Endpoints"
Cohesion: 0.5
Nodes (4): Approve or reject a pending cluster correction., review_feedback_endpoint(), Approve or reject a pending feedback correction.      - Approved corrections s, review_feedback()

### Community 18 - "Feedback Endpoints"
Cohesion: 0.5
Nodes (4): Submit a manual cluster correction for a requirement., submit_feedback_endpoint(), Submit a human-in-the-loop requirement cluster correction.      Updates the re, submit_feedback()

### Community 19 - "Community 19"
Cohesion: 0.67
Nodes (3): quality_history_endpoint(), Get the clustering-quality history across constrained iterations., get_quality_history()

## Knowledge Gaps
- **123 isolated node(s):** `Upload and preprocess a CSV or XLSX requirements file.`, `Run Phase 2 requirement enrichment and persist results for a session.`, `Run the clustering pipeline on an uploaded session.`, `Get pipeline progress for a session.`, `Get all clusters for a session.` (+118 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **9 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `generate_and_persist_suggestions()` connect `Refinement Service` to `Domain Embeddings & Ablation`, `Persistence & Phase-5 Services`, `Refinement LLM Provider`, `Embeddings & Representatives`, `Merge Suggestion`, `Split Suggestion`?**
  _High betweenness centrality (0.257) - this node is a cross-community bridge._
- **Why does `cluster_requirements_endpoint()` connect `Persistence & Phase-5 Services` to `LLM Provider & Enrichment`, `API Schemas & Routes`, `API Routes`, `Enrichment Persistence`?**
  _High betweenness centrality (0.108) - this node is a cross-community bridge._
- **Why does `_safe()` connect `Refinement Service` to `LLM Provider & Enrichment`?**
  _High betweenness centrality (0.092) - this node is a cross-community bridge._
- **Are the 32 inferred relationships involving `str` (e.g. with `upload_requirements()` and `cluster_requirements_endpoint()`) actually correct?**
  _`str` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 6 inferred relationships involving `normalize_plain_text()` (e.g. with `str` and `_safe_message()`) actually correct?**
  _`normalize_plain_text()` has 6 INFERRED edges - model-reasoned connections that need verification._
- **Are the 5 inferred relationships involving `run_and_persist_enrichment()` (e.g. with `enrich_requirements_endpoint()` and `normalize_plain_text()`) actually correct?**
  _`run_and_persist_enrichment()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `enrich_requirements()` (e.g. with `str` and `run_and_persist_enrichment()`) actually correct?**
  _`enrich_requirements()` has 2 INFERRED edges - model-reasoned connections that need verification._