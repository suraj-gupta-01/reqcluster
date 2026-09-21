# ReqCluster Refinement & Production Readiness Plan

**Scope:** clustering quality, refinement, human-in-the-loop, active learning, dependency engine, evaluation, and production hardening.  
**Basis:** current eval-branch architecture, the labeled BIT/control XLSX dataset, and the clustering PDF report.  
**Primary goal:** move ReqCluster from “impressive demo with weak evidence” to “analyst-trusted requirements copilot with governed production behavior.”

---

## Guiding principles

1. **Assisted authority, not autonomous authority.**  
   ReqCluster proposes; analysts approve for anything safety-significant.

2. **Evaluate against real labels before scaling claims.**  
   Generated UAV-FMS benchmarks are useful for stress, not sufficient for quality claims.

3. **Make every improvement measurable.**  
   Every phase has entry criteria, exit metrics, and rollback behavior.

4. **Preserve determinism and auditability.**  
   Same input + same config must produce comparable artifacts.

5. **Treat noise as a queue, not a trash can.**

---

## Phase 0 — Truth baseline and quick wins
**Duration:** 1–2 weeks  
**Goal:** stop overclaiming; establish trustworthy evaluation on the labeled dataset.

### Tasks
- Add `BIT_Control_Algorithm_Requirements.xlsx` to `eval/` as a gated dataset.
- Implement metrics: ARI, NMI, V-measure, purity, Hungarian-matched accuracy, noise rate by true group, confusion matrix.
- Add baselines: TF-IDF k-means, SBERT k-means with k=8, agglomerative, current UMAP+HDBSCAN.
- Add tiny-data clustering profile for `N < 500`.
- Run seed sweeps and record variance.
- Fix report language: replace “coverage” as a quality claim with correctness metrics.
- Publish a failure taxonomy: split group, merged group, wrong theme, noise omission, near-miss overlap.

### Exit criteria
- Labeled dataset metrics are reproducible in CI/eval harness.
- Current pipeline compared against at least 3 baselines.
- Known misclustered requirements from the BIT/control set are documented with reasons.

### Deliverables
- `eval/suites/bit_control.yaml`
- Updated metrics module
- Baseline comparison report
- Cluster failure taxonomy doc

---

## Phase 1 — Clustering quality for small/real datasets
**Duration:** 2–4 weeks  
**Goal:** make clustering good enough for analyst-assisted triage on real requirement sets.

### Tasks
- Add dataset-size-aware clustering configs:
  - `tiny`: `N < 500`
  - `small`: `500–5k`
  - `large`: `>5k`
- For tiny datasets, test SBERT+HDBSCAN without UMAP, low-dim UMAP, and agglomerative clustering.
- Tune HDBSCAN: `min_cluster_size`, `min_samples`, `cluster_selection_method`, `alpha`.
- Add optional constraint-aware clustering using approved must-link/cannot-link pairs.
- Add soft membership output: top-2/top-3 cluster probabilities where supported.
- Add overlap-aware evaluation if multi-label ground truth is created: Bcubed F1, Omega index, average label cardinality.
- Improve c-TF-IDF labeling stability: minimum docs per cluster, stopword handling, label confidence score separate from cluster quality.

### Exit criteria
- On BIT/control XLSX: material improvement over current ARI (~0.19–0.23). Target: **ARI ≥ 0.60** initially, then push toward **0.75+**.
- Noise rate on labeled dataset ≤ 10% with reviewed reasons.
- Seed variance documented; no unexplained nondeterminism.

### Deliverables
- Dataset-size clustering profiles
- Cluster config search tool
- Soft membership API/UI fields
- Updated clustering report with baselines and failures

---

## Phase 2 — Refinement engine that analysts actually trust
**Duration:** 3–5 weeks  
**Goal:** turn merge/split suggestions into a governed review workflow.

### Tasks
- Define suggestion types:
  - merge clusters
  - split cluster
  - move requirement
  - mark as overlap/secondary label
  - mark as noise with reason
- Score each suggestion with evidence:
  - semantic cohesion/separation
  - shared terms and EARS patterns
  - current constraint conflicts
  - impact on metrics if applied
- Add preview mode: before/after clusters, affected requirements, metric delta.
- Add bulk operations with dry-run.
- Add analyst decision capture: accepted/rejected, reason, role, timestamp.
- Feed accepted decisions into constraint pairs and re-clustering.

### Exit criteria
- Every refinement suggestion has evidence and impact preview.
- No destructive operation without confirmation.
- Suggestion acceptance/rejection history is auditable.
- Re-cluster after accepted feedback improves or maintains labeled metrics.

### Deliverables
- Refinement suggestion schema
- Preview/dry-run API
- Audit log UI/API
- Feedback-to-constraint pipeline

---

## Phase 3 — Human-in-the-loop and active learning hardening
**Duration:** 3–6 weeks  
**Goal:** make human feedback safe, useful, and measurable.

### Tasks
- Enforce role-based approval for feedback application.
- Separate feedback states: `submitted`, `reviewed`, `approved`, `applied`, `rejected`, `rolled_back`.
- Add risk tiers:
  - low: exact duplicates, obvious moves
  - medium: merge/split suggestions
  - high: cannot-link constraints, safety-significant groups
- Add uncertainty queue based on:
  - low membership probability
  - high inter-cluster similarity
  - disagreement between embedding and keyword evidence
  - active-learning acquisition score
- Add active-learning lift measurement:
  - metric improvement per accepted correction batch
  - acceptance rate by suggestion type
  - escaped errors after feedback
- Prevent feedback poisoning: limit auto-accept, require review for high-impact constraints, support rollback of constraint batches.

### Exit criteria
- All feedback actions are attributable and reversible.
- Active learning shows measurable lift on labeled dataset.
- No silent propagation of cannot-link constraints.

### Deliverables
- Feedback state machine
- Risk-tiered approval workflow
- Active-learning evaluation report
- Rollback mechanism for constraints and re-cluster iterations

---

## Phase 4 — Dependency engine improvement
**Duration:** 4–8 weeks  
**Goal:** improve realistic dependency recall while keeping precision under control.

### Priority order
1. **Pass 0 artifact extraction recall**
2. **Artifact synonym/near-match precision/recall**
3. **Sequential edge precision**
4. Hierarchical edge precision

### Tasks
- Build per-pass evaluation: explicit reference, artifact, sequential, hierarchical.
- Add failure classes: missed artifact, wrong producer/consumer, wrong direction, wrong relation label, spurious edge.
- Expand artifact extraction for EARS and aerospace/control vocabulary.
- Add domain lexicon/config per project: actuator, sensor, mode, BIT, fault, command, config, timing, diagnostic.
- Tune near-match threshold with precision/recall curve, not a single fixed 0.75.
- Tighten Pass 2 sequential rules:
  - require actor/object/state change
  - penalize pure reporting/diagnostic pairs
  - suppress edges when both requirements are same lifecycle/reporting type
- Keep evidence spans mandatory for UI display and export.

### Exit criteria
- Aerospace gold set detection F1 improves materially above current 0.387.
- Typed F1 improves without large spurious-edge increase.
- Per-pass metrics reported in eval harness.

### Deliverables
- Dependency pass-level scoring
- Artifact extraction improvement plan and implementation
- Edge explanation quality samples
- Updated dependency eval report

---

## Phase 5 — Production hardening
**Duration:** ongoing after Phase 2  
**Goal:** deploy as analyst copilot with governance.

### Must-have controls
- Deterministic runs: fixed seed/config, model version, dataset hash.
- Session/artifact versioning and rollback.
- Export labeling: inferred edges/clusters marked as inferred.
- Audit logs for upload, cluster, refine, feedback, re-cluster, export.
- Noise queue with aging and reason codes.
- Access control for approval actions.
- Optional external LLM disabled by default; explicit opt-in for external providers.
- CI gates for labeled datasets and dependency gold sets.

### Rollout path
1. **Shadow mode:** no authoritative writes.
2. **Assisted review:** analysts accept/reject suggestions.
3. **Limited auto-apply:** exact duplicates and explicit-reference edges only.
4. **Governed production:** all other actions require approval.

---

## Recommended immediate backlog

### P0 — before any further polish
- Add labeled BIT/control dataset to eval gates.
- Add confusion matrix and per-group metrics.
- Add tiny-data clustering profile.
- Replace PDF “coverage” framing with correctness metrics.
- Add rollback/versioning for clustering iterations.

### P1 — highest-value model work
- Constraint-aware re-clustering using approved feedback.
- Pass 0 artifact extraction recall improvements.
- Sequential dependency precision tightening.

### P2 — analyst trust
- Refinement preview/dry-run.
- Feedback state machine and approval workflow.
- Noise reason codes and queue.

### P3 — production evidence
- Shadow-mode deployment report.
- Acceptance rate of suggestions.
- Escaped-error analysis.
- Per-pass dependency metrics.

---

## Definition of success

ReqCluster is production-credible when all are true:

- It beats simple baselines on real labeled data.
- Analysts accept a high fraction of suggestions and catch the rest safely.
- Every output is explainable, versioned, and reversible.
- Noise is reviewed, not ignored.
- Dependency edges remain evidence-backed candidates until approved.
- Generated benchmark results are clearly separated from real dataset evidence.

## Definition of failure

- The system auto-merges/splits clusters without approval.
- The report claims quality from synthetic data only.
- Noise contains many requirements that obviously belong to known groups.
- Active learning reduces to “store user clicks” without measured lift.
- Dependency output is presented as truth despite low realistic recall.
