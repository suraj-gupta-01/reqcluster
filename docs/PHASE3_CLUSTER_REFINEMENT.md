# Phase 3 ClusterLLM Refinement

This document describes the Phase 3 (ClusterLLM Refinement) system architecture, algorithms, service API design, and frontend implementation details.

## 1. Algorithmic Foundation

Phase 3 introduces automated methods to analyze existing requirement clusters and identify opportunities for optimization (merging similar clusters or splitting heterogeneous clusters).

### Merge Suggestions
- **Centroid Computation**: Each cluster's centroid is calculated in the UMAP 10D embedding space.
- **Pairwise Similarity**: Pairwise cosine similarity is computed between all centroids.
- **Silhouette Delta**: If a merge is simulated:
  - The Silhouette Score is recalculated for the merged configuration.
  - The change in score (delta) determines if merging improves the overall structure.
- **Suggestion Criteria**: Clusters with high centroid similarity and positive/stable Silhouette Score changes are recommended for merging.

### Split Suggestions (Bimodality & Spread)
- **Gaussian Mixture Models (GMM)**: A GMM is fitted to each cluster in the embedding space (1D or 2D projection) to perform a bimodality test (comparing 1 vs 2 component fits).
- **Cluster Spread**: The standard deviation or variance of the embeddings within each cluster is monitored.
- **Suggestion Criteria**: Clusters with high variance and strong bimodal characteristics (indicating they consist of two distinct sub-groups of requirements) are recommended for splitting.

### Representative Requirements
- **Centroid-Nearest Extraction**: For each cluster, we extract the $K$ requirements (typically 3) whose embeddings are closest to the cluster's centroid.
- **Cluster Summary**: These representatives represent the core concept of the cluster and are used to build context for LLM coherence evaluation.

---

## 2. LLM Refinement Service

The LLM Refinement Service (`backend/llm_services/refinement.py` and `backend/services/refinement_service.py`) acts as the evaluator and advisor.

### Coherence Scoring
For each suggestion, the LLM analyzes:
- The title, description, and representative requirements of the cluster(s).
- Coherence Score: A rating from 1 to 5.
- Rationale: Detailed textual explanation of the coherence or lack thereof.

### Refinement Suggestions API
Suggestions are generated and saved to the database:
- **Type**: `merge` or `split`.
- **Target Clusters**: The ID(s) of the clusters involved.
- **Metrics**: Cosine similarity (for merge) or bimodality/spread (for split).
- **LLM Feedback**: Coherence score and textual rationale.
- **Status**: `pending`, `accepted`, or `rejected`.

---

## 3. Database & API Route Schema

### Database Models (`backend/models/database.py`)
- `RefinementSuggestion`:
  - `id`: Primary key (UUID/integer)
  - `suggestion_type`: `merge` or `split`
  - `source_cluster_id`: ID of the cluster to be split, or first cluster in merge
  - `target_cluster_id`: Second cluster in merge (null for split)
  - `similarity_score` / `bimodality_score`: Metric values
  - `coherence_score`: LLM coherence score (1-5)
  - `rationale`: LLM-provided rationale
  - `status`: `pending`, `accepted`, `rejected`
  - `created_at` / `updated_at`
- `RefinementAuditLog`:
  - `id`: Primary key
  - `action`: E.g., `generate_suggestions`, `apply_merge`, `apply_split`, `reject_suggestion`
  - `details`: JSON payload of what was changed and who performed it
  - `timestamp`

### API Endpoints (`backend/api/routes.py`)
- `POST /api/suggestions/generate`: Generates suggestions using backend ML metrics and LLM refinement provider.
- `GET /api/suggestions`: Lists active (pending, accepted, rejected) suggestions.
- `POST /api/suggestions/apply`: Applies a suggestion (mutates requirements and clusters table) and logs to the audit log.
- `GET /api/suggestions/audit`: Retrieves the audit history.

---

## 4. Frontend Refinement Interface

The refinement UI is integrated as a premium experience in the frontend dashboard:

- **Refinement Page (`frontend/src/pages/RefinementPage.jsx`)**: Displays suggestions organized by Merge and Split categories.
- **Suggestion Cards (`frontend/src/components/SuggestionCard.jsx`)**: Shows coherence scores, ML metrics, rationales, and provides single-click "Accept" or "Reject" buttons.
- **Comparison Panel (`frontend/src/components/ClusterComparisonPanel.jsx`)**: Visualizes representative requirements side-by-side to assist the user in reviewing merge or split candidates.
- **Audit Logs**: Provides a historical view of all refinement operations taken on the current workspace.
