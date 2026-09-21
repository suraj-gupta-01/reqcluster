# ReqCluster Refinement & Production Readiness Plan (Updated)

**Scope:** clustering quality, refinement, human-in-the-loop, active learning, dependency engine, evaluation, and production hardening.  
**Basis:** current `refine` branch architecture, labeled BIT/Control dataset, evaluation harness baseline, and codebase audit.  
**Primary goal:** move ReqCluster from “impressive demo with weak evidence” to “analyst-trusted requirements copilot with governed production behavior.”

---

## Codebase Audit Summary: What is Already Built vs. What Remains

Before prioritizing future work, an audit of the codebase reveals that significant architecture was already implemented or completed in Phase 0:

| Capability / Component | Codebase Status | Notes |
|---|---|---|
| **BIT/Control Dataset Wiring** | ✅ **Completed** | Converted to `data/bit_control_requirements.csv`, registered in `eval/datasets.yaml` with tolerances. |
| **Hungarian Accuracy & Confusion Matrix** | ✅ **Completed** | Implemented in `backend/core/metrics.py` (via `scipy.optimize.linear_sum_assignment`) and extracted in `eval/metrics.py`. |
| **Gated Metrics (ARI, NMI, V-Measure, Purity)** | ✅ **Completed** | Extracted and enforced in eval harness and drift tracking. |
| **Constraint-Aware Clustering** | ✅ **Already Built** | Implemented in `backend/core/constrained_clustering.py` + API route `/feedback/constraints` (must-link & cannot-link). |
| **Merge / Split Suggestions** | ✅ **Already Built** | Implemented in `backend/core/merge_suggest.py` and `backend/core/split_suggest.py`. |
| **Active Learning Core** | ✅ **Already Built** | Implemented in `backend/core/active_learning.py` (uncertainty sampling, margin sampling). |
| **Clustering Iteration Versioning** | ✅ **Already Built** | Schema present in `backend/models/database.py` (`ClusteringIteration` table). |
| **Audit Log Scaffolding** | ✅ **Already Built** | Schema present in `backend/models/database.py` (`ClusteringAuditLog` table with before/after state). |
| **Ablation Framework** | ✅ **Already Built** | Implemented in `backend/core/ablation.py`. |
| **Per-Pass Dependency Evaluation** | ✅ **Already Built** | Implemented in `backend/core/dependency_eval.py`. |
| **EARS Sequential Trigger Bound** | ✅ **Already Built** | Hardened in dependency engine (12-word trigger clause bound). |
| **Deterministic Seeds & Local LLM Opt-in** | ✅ **Already Built** | Fixed random seeds (`random_state=42`); mock LLM default with local OpenAI-compatible opt-in. |

---

## Guiding Principles

1. **Assisted authority, not autonomous authority.**  
   ReqCluster proposes; analysts approve for anything safety-significant.

2. **Evaluate against real labels before scaling claims.**  
   Generated UAV-FMS benchmarks are useful for stress, not sufficient for quality claims. Real aerospace datasets (e.g. BIT/Control) are the primary validation ground truth.

3. **Make every improvement measurable.**  
   Every phase has entry criteria, exit metrics, and rollback behavior.

4. **Preserve determinism and auditability.**  
   Same input + same config must produce comparable artifacts.

5. **Treat noise as a queue, not a trash can.**

---

## Phase 0 — Truth Baseline and Quick Wins
**Status:** **Core Complete** (Baselines & taxonomy documentation remaining)  
**Remaining Duration:** 2–3 days  
**Goal:** Establish trustworthy evaluation on real labeled datasets and eliminate unsubstantiated claims.

### Completed Tasks
- [x] Register `data/bit_control_requirements.csv` as `bit_control` in `eval/datasets.yaml` with gating tolerances.
- [x] Implement Hungarian-matched accuracy (`hungarian_accuracy_pct`) and confusion matrix generation (`confusion_matrix_rows`) in `backend/core/metrics.py` and `eval/metrics.py`.
- [x] Execute initial baseline run on `bit_control` and record in `eval/drift.json` (Result: N=100, 6 clusters, 42% noise, purity 37.93%, Hungarian accuracy 32.76%, ARI 0.0178).

### Remaining Tasks
- [ ] **Run baseline comparison suites on `bit_control`**: Execute `eval/baselines.py` (TF-IDF k-means, SBERT k-means k=8, agglomerative) to compare against UMAP+HDBSCAN.
- [ ] **Document failure taxonomy**: Document the known error categories observed in the BIT/Control baseline (e.g., 42% noise rate, split diagnostic groups, EARS-format confusion).
- [ ] **Fix report language**: Replace any remaining promotional "coverage" claims in export docs with verified correctness and purity metrics.

### Deliverables
- [x] `data/bit_control_requirements.csv` and updated `eval/datasets.yaml`
- [x] Updated `backend/core/metrics.py` and `eval/metrics.py`
- [ ] Baseline comparison report (`eval/results/baselines_bit_control.json`)
- [ ] `docs/cluster_failure_taxonomy.md`

---

## Phase 1 — Clustering Quality for Small / Real Datasets
**Status:** **Pending** (Constraint-aware clustering already built)  
**Estimated Duration:** 2–3 weeks  
**Goal:** Optimize clustering for real engineering requirement sets where N < 500 without overfitting to synthetic distributions.

### Completed Prerequisite Tasks
- [x] Constraint-aware clustering engine (`backend/core/constrained_clustering.py`).
- [x] Basic HDBSCAN soft-membership extraction (`probabilities_` returned by model).

### Remaining Tasks
- [ ] **Dataset-size-aware clustering profiles**:
  - `tiny` (`N < 500`): Direct SBERT + HDBSCAN without UMAP, low-dimensional UMAP (2–5 dims), and Agglomerative / Ward fallback paths.
  - `small` (`500 <= N <= 5000`): Standard UMAP + HDBSCAN with tuned min_cluster_size.
  - `large` (`N > 5000`): Fast approximation / sub-sampling.
- [ ] **HDBSCAN parameter tuning for sparse engineering text**:
  - Tune `min_cluster_size`, `min_samples`, `cluster_selection_method` (`leaf` vs `eom`), and metric distance functions.
- [ ] **Expose soft membership in API & UI**:
  - Surface top-2 and top-3 alternative cluster memberships and probabilities in API responses for boundary requirements.
- [ ] **Decouple c-TF-IDF labeling confidence from cluster silhouette**:
  - Calculate distinct lexical label confidence scores based on term discriminability and cluster size.
- [ ] **Multi-label overlap evaluation** (if secondary labels assigned):
  - Support Bcubed F1 or Omega index for cross-subsystem requirements.

### Exit Criteria
- Measurable improvement on BIT/Control dataset over baseline ARI (0.018 → target ARI ≥ 0.40 initial, stretching toward 0.60).
- Unassigned noise rate reduced from 42% to ≤ 20% on the BIT/Control dataset.
- Seed variance verified and documented.

---

## Phase 2 — Refinement Engine That Analysts Actually Trust
**Status:** **Pending** (Core suggestion algorithms already exist)  
**Estimated Duration:** 3–4 weeks  
**Goal:** Turn algorithmic suggestions into an auditable, interactive review workflow.

### Completed Prerequisite Tasks
- [x] Cluster merge suggestion heuristics (`backend/core/merge_suggest.py`).
- [x] Cluster split suggestion heuristics (`backend/core/split_suggest.py`).
- [x] `ClusteringAuditLog` database schema (`backend/models/database.py`).

### Remaining Tasks
- [ ] **Transactional Preview / Dry-Run Mode**:
  - Create `/api/refine/preview` endpoint calculating before/after cluster metrics, affected requirements, and delta metrics without persisting state to DB.
- [ ] **Structured Suggestion Evidence Scoring**:
  - Standardize suggestion schema with explicit evidence: semantic cohesion/separation distance, shared EARS/domain keywords, and constraint conflict checks.
- [ ] **Analyst Decision Capture with Reason Enums**:
  - Extend feedback capture with structured decision reasons: `wrong_subsystem`, `too_broad`, `too_specific`, `duplicate`, `spurious_merge`.
- [ ] **Bulk Operations with Validation**:
  - Support batched merge/split approvals with pre-flight dry-run checks.

### Exit Criteria
- Every merge/split suggestion provides structured evidence and zero-write preview.
- No irreversible clustering modification can be made without explicit user confirmation.
- Complete auditability of accepted vs. rejected suggestions.

---

## Phase 3 — Human-in-the-Loop & Active Learning Hardening
**Status:** **Pending** (Basic models and sampling algorithms already exist)  
**Estimated Duration:** 3–5 weeks  
**Goal:** Make human corrections governable, secure against poisoning, and measurable in their impact.

### Completed Prerequisite Tasks
- [x] `FeedbackCorrection` database table (`backend/models/database.py`).
- [x] Uncertainty and margin sampling algorithms (`backend/core/active_learning.py`).

### Remaining Tasks
- [ ] **Extend Feedback State Machine**:
  - Migrate status enum from `pending/approved/rejected` to:  
    `submitted` → `reviewed` → `approved` → `applied` → `rejected` → `rolled_back`.
- [ ] **Constraint Batch Rollback**:
  - Implement an `/api/feedback/rollback` endpoint leveraging `ClusteringIteration` to restore cluster state before a feedback batch was applied.
- [ ] **Risk-Tiered Approval Workflow**:
  - Classify actions by risk:
    - *Low*: Exact duplicate resolution, single-item moves.
    - *Medium*: Multi-item cluster splits and merges.
    - *High*: Global cannot-link constraints affecting safety-critical subsystems.
- [ ] **Uncertainty Queue Integration**:
  - Connect `active_learning.py` uncertainty scores to the analyst triage queue (prioritizing low membership probability and conflicting embedding/lexical evidence).
- [ ] **Measure Active Learning Lift**:
  - Automated evaluation of metric gain (ΔARI, ΔHungarian Accuracy) per batch of human feedback applied.

### Exit Criteria
- All human corrections are completely reversible with one-click batch rollback.
- Measurable positive lift in ARI on benchmark sets when simulating analyst feedback.
- Strict prevention of unchecked cannot-link propagation.

---

## Phase 4 — Dependency Engine Improvement
**Status:** **Pending** (Evaluation harness and trigger bounds already built)  
**Estimated Duration:** 4–6 weeks  
**Goal:** Increase realistic dependency recall and typed accuracy while suppressing spurious edges.

### Completed Prerequisite Tasks
- [x] Pass-by-pass dependency evaluation engine (`backend/core/dependency_eval.py`).
- [x] Mandatory evidence spans for all extracted edges.
- [x] EARS sequential trigger clause 12-word bound.

### Remaining Tasks
- [ ] **Domain Lexicon & Project-Specific Configuration**:
  - Move beyond hardcoded verbs to configurable project-level lexicons (e.g. aerospace telemetry, BIT, actuators, sensors, fault conditions, commands).
- [ ] **Near-Match Artifact Threshold Optimization**:
  - Replace static `0.75` similarity threshold with a precision/recall calibration curve tailored to aerospace terminology.
- [ ] **Sequential Pass 2 Precision Tightening**:
  - Enforce actor/state changes; suppress sequential links between pure telemetry/status reporting requirements; penalize same-lifecycle duplicate status statements.
- [ ] **Structured Dependency Failure Reporting**:
  - Categorize errors in eval outputs: `missed_artifact`, `wrong_producer_consumer`, `wrong_direction`, `spurious_relation`.

### Exit Criteria
- Aerospace gold set detection F1 materially exceeds baseline (0.387 → target F1 ≥ 0.55).
- Spurious sequential edges between purely descriptive requirements reduced by ≥ 50%.
- Pass-level metrics integrated into standard CI eval report.

---

## Phase 5 — Production Hardening & Governance
**Status:** **Partially Implemented**  
**Estimated Duration:** 2–3 weeks (Parallel with Phase 2/3)  
**Goal:** Ensure enterprise auditability, safety controls, and safe deployment modes.

### Completed Prerequisite Tasks
- [x] Deterministic execution controls (fixed random seed, config version tracking).
- [x] Local LLM opt-in (external API providers disabled by default).
- [x] CI eval drift gating (`eval/drift.json`, `eval/datasets.yaml`).

### Remaining Tasks
- [ ] **Export Data Governance**:
  - Explicitly flag all exported edges and clusters with provenance (`human_verified` vs. `inferred_algorithmic`).
- [ ] **Dedicated Noise Queue & Aging**:
  - Provide a dedicated UI/API view for noise elements with reason classification (`insufficient_context`, `cross_cutting`, `malformed_spec`).
- [ ] **Role-Based Access Control (RBAC)**:
  - Add authentication and authorization checks to approval and rollback endpoints.
- [ ] **Shadow Mode Deployment Harness**:
  - Automated logging mode to compare live analyst workflows against ReqCluster recommendations without applying changes.

---

## Realistic Timeline & Execution Roadmap

| Phase | Focus Area | Status | Realistic Estimate | Priority |
|---|---|---|---|---|
| **Phase 0** | Truth Baseline & Metrics | **80% Complete** | **2–3 days** | P0 (Immediate) |
| **Phase 1** | Small-Data Clustering Quality | Not Started | **2–3 weeks** | P1 |
| **Phase 2** | Refinement Preview & Dry-Run | Not Started | **3–4 weeks** | P2 |
| **Phase 3** | Human-in-the-Loop State Machine | Not Started | **3–5 weeks** | P3 |
| **Phase 4** | Dependency Engine & Lexicons | Not Started | **4–6 weeks** | P1 (High Impact) |
| **Phase 5** | Governance, RBAC & Export Flags | Partially Built | **2–3 weeks** | P2 |

**Recommended Immediate Next Sprint (1–2 Weeks):**
1. Finish Phase 0 remaining items (run baseline comparison on BIT/Control, generate failure taxonomy doc).
2. Implement Phase 1 `tiny` clustering profile (`N < 500` without UMAP / agglomerative fallback) to immediately fix the 42% noise rate and low ARI on the BIT/Control requirements.
3. Add project-level domain lexicon overrides for Phase 4 dependency extraction.
