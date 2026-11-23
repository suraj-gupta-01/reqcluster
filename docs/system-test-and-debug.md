# System test and debug report

Date: 2026-11-23  
Baseline revision: `1c75b99` (`refine`, after the `origin/overhaul` fast-forward)  
Scope: investigate clustering quality, report accuracy and related evaluation issues. Subsequent fixes are recorded below; the other findings remain open unless stated otherwise.

**Repository state at documentation commit:** The fixes and regression tests listed below were verified in the working tree but were not included in the documentation-only commit. Apply or commit those code changes separately before relying on their fix status in another checkout.

## Fix status

| Finding | Status | Change and verification |
|---|---|---|
| 1. Pre-clustering PDF coverage | **Completed** | `backend/reqcluster/export/pdf.py` counts only assigned, non-noise requirements. Regression tests cover unclustered and mixed assignments; both failed before the fix. |
| 2. Single-cluster Hungarian accuracy | **Completed** | `backend/reqcluster/ml/metrics.py` computes the optimal match for one cluster instead of returning zero. Regression test failed before the fix. |
| 8. Provider-insensitive explanation cache | **Completed** | `backend/reqcluster/services/dependencies.py` checks the requested or configured provider before accepting a cached explanation. |
| 9. Explanations after tree regeneration | **Completed for sequential regeneration** | Saving a new dependency tree clears its run's explanation cache in the same transaction. An explanation already in flight during regeneration has not been tested; cache provenance still does not distinguish model changes within one provider. |

After these fixes, `uv run --no-sync pytest tests/unit/test_metrics.py tests/unit/test_export.py tests/eval -q` passed **36 tests**, `uv run --no-sync pytest -q` passed **338 tests** (175 deselected), `uv run --no-sync pytest -m acceptance tests/acceptance -q` passed **104 tests**, and `uv run --no-sync ruff check backend tests eval scripts` passed. Existing reports and experiments below describe the behavior observed before these changes.

After the dependency-cache fixes, `uv run --no-sync pytest tests/unit/test_svc_dependencies.py -q` passed **5 tests**, `uv run --no-sync pytest -q` passed **340 tests** (175 deselected), the acceptance suite passed **104 tests**, and Ruff passed.

## Executive summary

- The supplied PDFs confirm that the old run had 10 noise requirements, while the new enriched HDBSCAN and Ward-with-outliers reports each had 18. The new reports leave all four PID requirements `REQ_046`–`REQ_049` as noise; the old report clustered three of them.
- On the labeled 100-requirement workbook, a separate **base-text**, CPU MiniLM experiment across 20 seeds produced 0–22 noise requirements and ARI 0.132–0.240. This demonstrates seed sensitivity in that experimental setup, not nondeterminism at a fixed seed or the cause of the enriched reports.
- Repeating the complete reduction and HDBSCAN calculation at seed 0 with the same embeddings in the same process produced identical labels.
- Two additional reporting defects were reproduced and subsequently fixed: an unclustered PDF reported 100% coverage, and one-cluster Hungarian accuracy was reported as zero even when 50% of items could be matched.
- Ward-with-outliers keeps HDBSCAN's noise mask and returns membership values of 1.0 for all assigned requirements. These values should not be interpreted as calibrated confidence.

## Inputs and comparison

| Artifact | Description |
|---|---|
| `data/bit_control_requirements.xlsx` | 100 requirements, eight human-assigned groups, `Requirements` sheet. |
| `reqcluster_session_15 (1).pdf` | User-identified old architecture report. |
| `session-1.pdf` | User-identified new architecture, enriched HDBSCAN. |
| `session-1 (1).pdf` | User-identified new architecture, enriched Ward plus HDBSCAN outliers. |

PDF cluster memberships were manually transcribed from the readable report text, checked for duplicate IDs and joined by numeric requirement ID to the workbook. All 100 IDs were accounted for. ARI and NMI were recomputed using scikit-learn. For the all-items metrics, noise is treated as one predicted label; for assigned-only metrics, noise rows are excluded.

| Report | Clusters | Noise | ARI all | ARI assigned | NMI all | Purity assigned |
|---|---:|---:|---:|---:|---:|---:|
| Old | 6 | 10 | 0.192 | 0.228 | 0.380 | 53.3% |
| Enriched HDBSCAN | 7 | 18 | 0.195 | 0.244 | 0.406 | 57.3% |
| Enriched Ward + outliers | 6 | 18 | 0.187 | 0.227 | 0.372 | 52.4% |

Purity is the sum of each non-noise cluster's most frequent ground-truth group divided by the number assigned. It permits several clusters to map to the same group; it is not one-to-one accuracy. Noise-excluded metrics should always be read alongside noise rate.

The Command and Output group has 10 workbook entries. The old PDF assigned 9 to a cluster; each new PDF assigned 4. This measures assignment coverage, **not** correct-theme recall. Both new PDFs have the same 18-item noise set: `009, 028, 029, 035, 046, 047, 048, 049, 063, 072, 074, 075, 078, 079, 080, 085, 088, 092`.

The prior claim of 38 changed assignments is not established by a raw cluster-ID comparison: cluster IDs are arbitrary. The new reports use different clustering options. The Ward report merges the HDBSCAN sensor cluster into its larger BIT-labelled cluster, and also moves `REQ_010`, `055`, `057`, and `060` relative to their previous groups.

## Experiment A: 20 seeds, base text, CPU

**Purpose:** assess sensitivity to the random seed, holding embeddings and algorithm settings fixed.

- Input: `data/bit_control_requirements.xlsx`, raw requirement text, workbook group as ground truth.
- Model: cached `all-MiniLM-L6-v2`, CPU, 384 dimensions, normalized vectors; offline mode. These are **base embeddings**, not the enriched inputs from the PDFs.
- Reduction: `backend/reqcluster/ml/reduce.py`: 10-dimensional UMAP, cosine metric, 15 neighbors, min_dist 0, 200 epochs, single UMAP worker; seeds 0–19.
- Clustering: scikit-learn HDBSCAN through `cluster.hdbscan`, `min_cluster_size=5`, `min_samples=3`, EOM. Noise is predicted label `-1`.
- All-item ARI/NMI: `sklearn.metrics.adjusted_rand_score` and `normalized_mutual_info_score`, with noise treated as one cluster. Population standard deviation shown below.

| Seed | Noise | Clusters | ARI all | NMI all |
|---:|---:|---:|---:|---:|
| 0 | 13 | 9 | 0.1671 | 0.4054 |
| 1 | 8 | 7 | 0.2328 | 0.4108 |
| 2 | 0 | 6 | 0.1316 | 0.3251 |
| 3 | 9 | 8 | 0.1927 | 0.4141 |
| 4 | 1 | 3 | 0.1374 | 0.3201 |
| 5 | 8 | 8 | 0.2371 | 0.4509 |
| 6 | 8 | 8 | 0.2206 | 0.4299 |
| 7 | 4 | 7 | 0.2356 | 0.4021 |
| 8 | 9 | 8 | 0.1957 | 0.4190 |
| 9 | 7 | 8 | 0.2399 | 0.4186 |
| 10 | 22 | 9 | 0.1593 | 0.3995 |
| 11 | 8 | 6 | 0.2287 | 0.3821 |
| 12 | 10 | 6 | 0.1766 | 0.3833 |
| 13 | 17 | 8 | 0.1726 | 0.3908 |
| 14 | 11 | 7 | 0.2059 | 0.4075 |
| 15 | 6 | 8 | 0.1966 | 0.4061 |
| 16 | 8 | 8 | 0.2067 | 0.4340 |
| 17 | 9 | 9 | 0.1536 | 0.3946 |
| 18 | 19 | 6 | 0.1918 | 0.3805 |
| 19 | 8 | 7 | 0.1724 | 0.3687 |

| Measure | Mean | Population SD | Range |
|---|---:|---:|---:|
| Noise count | 9.25 | 5.20 | 0–22 |
| ARI all | 0.1927 | 0.0326 | 0.1316–0.2399 |
| NMI all | 0.3972 | 0.0314 | 0.3201–0.4509 |
| Cluster count | 7.30 | 1.38 | 3–9 |

The repeat of seed 0 started from the same saved embeddings and independently reran both UMAP and HDBSCAN: exact labels matched, adjusted Rand between results 1.0. This is one repeat on one machine and does not establish cross-device determinism.

**Interpretation:** A no-noise result does not guarantee improved group agreement: seed 2 produced zero noise but ARI 0.1316. The test does not establish why the enriched reports had 18 noise points, nor whether enrichment helped or hurt: enriched text, model revision, effective seed and device for those PDFs were not reconstructed.

## Experiment B: parameter sensitivity

Using the same base embeddings and seed-0 UMAP projection, vary HDBSCAN only:

| Minimum cluster size | Minimum samples | Noise | Clusters | ARI all | Command/Output assigned |
|---:|---:|---:|---:|---:|---:|
| 3 | 1 | 3 | 12 | 0.1633 | 10/10 |
| 3 | 3 | 9 | 11 | 0.1498 | 10/10 |
| 3 | 5 | 16 | 8 | 0.1724 | 6/10 |
| 5 | 1 | 3 | 10 | 0.1720 | 10/10 |
| 5 | 3 | 13 | 9 | 0.1671 | 9/10 |
| 5 | 5 | 16 | 7 | 0.1617 | 6/10 |
| 8 | 1 | 6 | 6 | 0.2022 | 9/10 |
| 8 | 3 | 1 | 5 | 0.1768 | 10/10 |
| 8 | 5 | 22 | 5 | 0.1632 | 5/10 |

With `min_samples=2`, results for each of the three cluster sizes matched the corresponding `min_samples=1` results in this particular experiment. These are exploratory, in-sample results on one projection, not validated recommended defaults. Assigning all 10 output items is not the same as assigning them to an appropriate group.

## Experiment C: Ward versus HDBSCAN

Same base embeddings and projection for each seed; `min_cluster_size=5`, `min_samples=3`. Results from `pipeline._cluster`:

| Seed | HDBSCAN noise / clusters / ARI | Ward noise / clusters / ARI |
|---:|---|---|
| 0 | 13 / 9 / 0.1671 | 13 / 16 / 0.1669 |
| 1 | 8 / 7 / 0.2328 | 8 / 12 / 0.2006 |
| 2 | 0 / 6 / 0.1316 | 0 / 16 / 0.1830 |
| 3 | 9 / 8 / 0.1927 | 9 / 8 / 0.1884 |
| 4 | 1 / 3 / 0.1374 | 1 / 10 / 0.2135 |

`backend/reqcluster/ml/pipeline.py:11–20` computes HDBSCAN first and masks Ward labels with HDBSCAN's noise. Ward does not independently recover outliers. A synthetic two-group experiment returned only probability value `1.0` for all assigned Ward items: the code explicitly generates 1.0 for every assigned row and 0.0 for noise. These are implementation placeholders, not evidence that all Ward memberships are equally certain.

## Additional reproduced findings

### 1. PDF says 100% coverage before clustering — completed

- **Reproduction:** with a temporary isolated application home and mock providers, upload a two-row CSV via `POST /api/sessions`; without starting clustering, request `GET /api/sessions/{sid}/export/pdf`. The endpoint returned HTTP 200 while session status was `uploaded`.
- A separate direct PDF exporter reproduction using 30 `ExportReq` entries with `cluster_id=None`, `is_noise=False`, and zero clusters generated: `30 requirements, 0 clusters, 0 noise, 100.0% coverage`.
- **Cause in code:** `backend/reqcluster/export/bundle.py:63–84` represents unassigned requirements as `cluster_id=None` and `is_noise=False`; `backend/reqcluster/export/pdf.py:80–93` defines coverage as `(n - noise) / n`. Unassigned is therefore counted as covered.
- The exporter reproduction and endpoint reachability were checked separately; no PDF text extraction library was installed for the endpoint response. The frontend overview uses the same `total - noise` expression (`frontend/src/pages/OverviewPage.jsx:112–140`), but its pre-clustering display was not browser-tested.
- **Impact:** misleading report, especially when no clustering has happened.

### 2. Hungarian accuracy is zero for a single predicted cluster — completed

- **Reproduction:** `compute(np.array([0,0,0,0]), None, 42, ['A','A','B','B'])`.
- **Observed:** purity 50%, Hungarian accuracy 0%, noise-inclusive accuracy 0%, confusion `[[2,2]]`. A maximum one-to-one match can correctly match 2 of 4, or 50%.
- **Cause in code:** `backend/reqcluster/ml/metrics.py:23–27` explicitly returns zero when fewer than two predicted clusters exist. This is not the mathematical maximum assignment count.
- **Impact:** misleading accuracy for degenerate one-cluster outputs. The current code intentionally implements the shortcut; whether consumers depend on that convention requires review.

### 3. Noise can inflate reported quality when ignored

- **Reproduction:** predicted labels `[0,0,1,1,-1,-1,-1,-1]`, truth `['A','A','B','B','A','A','B','B']`.
- **Observed:** 50% noise and 50% coverage, but ARI, NMI, purity and Hungarian accuracy all equal 1.0 on assigned rows; noise-inclusive Hungarian accuracy is 0.5.
- **Cause:** `backend/reqcluster/ml/metrics.py:61–83` masks noise for the extrinsic metrics and confusion matrix. `eval/metrics.py:12–19` calls purity `accuracy_pct` for backward compatibility. This is defined behavior, but the denominator and convention matter when presenting results.
- **Impact:** a quality gate focused on assigned-only metrics can mask degraded coverage. The `bit_control` suite's configured noise ceiling is 60% (`eval/datasets.yaml:2–20`), not a domain-level guarantee.

### 4. PDF listings truncate each group, including noise

- **Reproduction:** generate a PDF bundle with 30 noise items; only 25 requirement entries were rendered, followed by `... and 5 more.` The same applies to 30 unclustered items.
- **Cause:** `backend/reqcluster/export/pdf.py:21,111–125` caps **each** listing at 25, including noise. The omission is disclosed in the PDF, but individual IDs beyond the cap cannot be reconstructed from it.
- **Impact:** PDF-only comparisons on larger reports can silently become incomplete if the omission line is overlooked. Use the CSV export for a complete assignment comparison.

### 5. Run provenance is incomplete in PDF and effective seed can be implicit

`backend/reqcluster/services/clustering.py:102–113` uses the global configured seed when the request seed is `None`, but stores the request parameters including that `None`; it also stores the embedding model name. The PDF export bundle omits run parameters and the PDF does not show model, effective seed, embedding mode, device or HDBSCAN/UMAP settings. Therefore the three PDFs alone cannot establish configuration equivalence. The stored parameter may be enough when seed is explicit; an implicit seed cannot be recovered from `params` alone if the global setting later changes.

### 6. Model revision is not pinned by the embedder

`SbertEmbedder` records the configured `revision`, and the cache key includes it, but the `SentenceTransformer` construction at `backend/reqcluster/embeddings.py:47–60` does not pass the revision argument to the model loader. Thus the configured revision is not demonstrated to control the downloaded weights. This is a code inspection finding; no network/model revision comparison was run.

## Existing suite and provider results from this session

These were run before the experiments above, on this same revision:

| Check | Result |
|---|---|
| Default pytest | 335 passed, 175 deselected, 65.26 s |
| Acceptance | 104 passed, 80.88 s |
| Hardening, not slow | 68 passed, 3 deselected, 80.98 s |
| Hardening, slow | 3 passed, 68 deselected, 72.94 s |
| Ruff backend/tests/eval/scripts | Passed |
| Frontend lint and production build | Passed; build warned about stale Browserslist data and large Plotly bundle |

The default and hardening suites required a process-only `REQCLUSTER_LLM_PROVIDER=mock` override at the time because the earlier `.env` value `openai-compatible` was invalid. The `.env` was subsequently changed at the user's request to point to LM Studio.

Real LM Studio smoke: Qwen2.5 3B, 20 requirements, mock embeddings: enrichment job succeeded in 476.5 s with 20 reported succeeded and one fallback; dependency generation succeeded with five edges; one LLM explanation succeeded. A refinement job succeeded but no generated suggestion was printed. This was not a full real-embedding quality test.

## Follow-up investigation: feedback and dependency workflows

These checks were performed in separate temporary application homes with 30 synthetic requirements, the mock embedding and LLM providers, synchronous jobs (`REQCLUSTER_JOB_WORKERS=0`), and spaCy disabled. They did not touch existing user sessions. They exercise the real service/database paths; they are not browser tests or real-model quality tests.

The existing focused suite `uv run --no-sync pytest tests/unit/test_svc_feedback.py tests/unit/test_svc_dependencies.py -q` subsequently passed: **14 passed in 17.74 seconds**. Passing existing tests does not invalidate the separate reproductions below; those scenarios were not asserted in that run.

### 7. Pending feedback is applied and carried forward before review

- **Reproduction:** cluster the synthetic session, move a requirement from cluster 0 to cluster 1 with `feedback.submit`, and inspect its result and constraints. The result was `status=pending`, with six constraints already created. Call `feedback.recluster` without approving it: the job succeeded and the six feedback-linked constraints appeared on the new run.
- Attempting `feedback.review(sid, original_feedback_id, False)` after reclustering raised `NotFound`. The new run's feedback queue was empty, while its constraint list still contained six entries referring to the original feedback ID.
- **Trace:** `services/feedback.py:128–149` immediately changes the assignment and creates constraints; `:245–274` carries all run constraints into the next run; `:163–169` reviews only feedback belonging to the current run.
- **Impact:** a pending correction can influence a new run and become inaccessible to the current-run review workflow while its constraints persist. Immediate application of a pending correction may be intended product policy; however, the loss of a straightforward review/reject path after reclustering is directly observed. No claim is made that the constraints changed additional assignments in this particular reproduction: the recluster job reported `moved=0`.

### 8. Dependency explanation cache ignores the requested provider — completed

- **Reproduction:** generate a dependency tree, call `dependencies.explain` for an edge with `provider='mock'`, then call the same edge with `provider='openai'` and `force=False`. The first result was `provider=mock, cached=False`; the second was `provider=mock, cached=True`. No OpenAI-compatible request was attempted for the second call.
- **Trace:** `services/dependencies.py:134–147` reads the cached explanation by `(run_id, source, target)` before accounting for the requested provider. A forced request bypasses that read.
- **Impact:** a user selecting a different provider can receive an older explanation from another provider. The response includes its actual provider and cache flag, so a caller can detect the mismatch if it checks them.
- **Fix verification:** the provider-specific regression test failed before the fix and passed after it; the five-test `tests/unit/test_svc_dependencies.py` suite passed in 15.76 seconds.

### 9. An explanation can outlive changed edge evidence — sequential regeneration fixed

- **Reproduction:** after caching a mock explanation, directly alter the persisted test tree's first edge from `hierarchical` to `data` and change its rationale. Requesting the same explanation again returned `cached=True` and exactly the original text.
- **Trace:** `services/dependencies.py:112–118` replaces a generated tree without clearing explanation rows; `:134–147` checks only the run and endpoint pair, not relation or evidence. The direct database alteration isolates the cache behavior; this experiment did **not** independently demonstrate that a normal regeneration changed that edge.
- **Impact:** if tree regeneration changes the relation or evidence for an unchanged pair, an old cached explanation can describe the previous edge. A forced explanation is a workaround, not a verification of correctness.
- **Fix verification:** a regression test caches an explanation, regenerates the tree, and confirms the next request misses the per-edge cache. It failed before the fix and passed after it. Concurrent in-flight explanations and model changes behind the same provider remain untested.

### 10. Stale dependency edges remain in export bundles

- **Reproduction:** generate a dependency tree for the synthetic session: 29 edges and `stale=False`. Move a requirement to another cluster through feedback. `dependencies.get(sid)` then reported `stale=True`; `export.bundle.build(sid)` still included all 29 edges and the `ExportBundle` had no stale-status field.
- **Trace:** staleness compares the tree's recorded edit sequence with the run's current edit sequence (`services/dependencies.py:122–131`). `export/bundle.py:86–92` reads the tree edges without that check. SysML/XMI and Jama exporters consume `b.edges`.
- **Impact:** downstream exports can contain dependency edges generated for an older cluster state without an export-level indication that the tree is stale. The reproduction establishes stale provenance, not that any of those specific 29 edges became factually incorrect.

### 11. Moving a requirement to noise stores membership 1.0

- **Reproduction:** submit feedback moving a clustered requirement to `new_cluster_id=None`. Reading it back returned `cluster_id=-1`, `is_noise=True`, and `probability=1.0`.
- **Trace:** `services/assignments.py:27–33` writes `probability=1.0` for every manual reassignment, including the noise cluster.
- **Impact:** the stored number is a human-decision marker in the current design, not a meaningful probability of membership in a normal cluster; downstream API consumers must not interpret it as a model confidence for noise. UI rendering of this exact case was not tested.

### 12. Active-learning ranking does not use membership probability

- **Reproduction:** for a fixed four-item, two-cluster geometry, calling `ml.active.uncertainty` once with probabilities `[.01,.02,.03,.04]` and again with `[.99,.98,.97,.96]` produced identical ranking and uncertainty scores; only the returned `membership_prob` fields changed.
- **Trace:** `ml/active.py:17–50` ranks by centroid margin, not the probability input. This is explicitly documented in the helper and is therefore not automatically a code defect.
- **Impact:** the UI or external users should not assume that low HDBSCAN membership probability directly raises active-learning priority. This distinction also matters because Ward's assigned probabilities are all 1.0.

### Constraint sanity check

As a limited negative check, 10,000 generated eight-item cases with random cannot-link pairs were passed through `ml.constraints.apply`; none left a tested cannot-link pair in the same non-noise cluster. This is not exhaustive proof, and did not cover must-link components or transitive conflicts.

## Architecture review

This review identifies demonstrated issues separately from design risks. It does not assert that every risk affects the supported local single-process deployment.

### 1. Run provenance is not an immutable snapshot — high

`services/clustering.py:62–75` reconstructs embeddings from the session's current enrichment and the currently configured embedder. Enrichment rows can be updated (`services/enrichment.py:99–124`), while the run stores requested parameters and a model name, not a frozen copy of every enriched input, embedding model revision, device and effective seed.

**Risk:** subsequent refinement, coherence or dependency analysis may operate on inputs different from those that produced the saved assignments. The effect on the supplied PDF runs has not been independently reproduced. See also findings 5 and 6 above.

### 2. Feedback approval and run boundaries conflict — high

Pending corrections immediately create constraints and assignments, while reclustering creates a new run and the review service only accepts current-run feedback. The temporary-session reproduction in finding 7 showed pending constraints carried into a new run while the original feedback became unreviewable through that service.

**Design question:** define whether pending corrections should be applied, and ensure their review state and constraints retain coherent ownership across revisions.

### 3. Artifact freshness is not consistently propagated — high

Dependency trees expose a stale flag based on the run's edit sequence, but export bundles still include stale tree edges without exposing that status (finding 10). Graph edges are stored separately in run NPZ files while assignments can change.

**Risk:** views or exports may combine current assignments with artifacts produced against an earlier state. The stale dependency export provenance was reproduced; whether an individual exported edge is actually incorrect depends on the change.

### 4. Job execution assumes a single application process — high for multi-process deployment

`jobs.py:61–68` executes work in process-local thread pools; `jobs.py:177–187` marks every queued or running job failed at startup. Starting another backend process against the same database could therefore mark jobs belonging to an existing live process as interrupted. A crash also loses in-memory work despite persistent job records.

This is a code-based scaling risk, not a reproduced multi-process failure. It need not block a deliberately single-instance local deployment.

### 5. Database records and run files have separate storage lifecycles — medium

`services/clustering.py:144–185` persists SQL state and run arrays in separate NPZ files. The implementation handles ordinary transaction exceptions and uses atomic file replacement, but SQL and filesystem writes do not form a single atomic transaction.

**Risk:** a hard process or storage failure may leave orphaned files or unavailable run data. Backups and deployments need to preserve both the database and run directory consistently. No crash-injection experiment was performed for this review.

### 6. A single cluster assignment may not reflect overlapping responsibilities — medium

Each requirement has one `cluster_id` in `models.py:82–92`; `-1` represents noise. Some requirements in the labeled workbook plausibly span multiple engineering themes, such as fail-safe output or actuator feedback.

This is a conceptual limitation rather than proof that a multi-label approach would perform better. Noise items currently have no built-in top-two candidate assignment in the report.

### 7. Evaluation can favor assigned items while hiding rejection — medium

`ml/metrics.py:61–83` computes several extrinsic scores after excluding noise; `eval/metrics.py:11–15` maps purity to the legacy name `accuracy_pct`. Finding 3 demonstrates that assigned-only scores can be perfect while half of the items are noise.

Noise and noise-inclusive metrics are also available, so the issue is how metrics are gated and communicated rather than a claim that all evaluations are wrong. Per-group recall and explicit noise treatment are important for this use case.

### 8. Security boundary is a trusted deployment, not multi-tenant isolation — deployment-dependent

`app.py:46–48, 82–84` implements an optional shared API key; an empty key disables that check. Session and requirement models do not contain user or tenant ownership (`models.py:39–58`).

This may be appropriate for the current local setup. It does not itself provide per-user authorization or tenant isolation for a shared service. No security penetration test was performed.

### 9. Schema upgrades require an operator decision — medium operational risk

`db.py:81–109` checks for missing model columns and fails startup rather than migrating old databases; it tells the operator to move the database aside or use a fresh one. This protects against silently running an incompatible schema but does not provide an automatic data-preserving upgrade path.

## Reproduction notes

Dependencies: `uv sync --locked --extra all --extra cpu`. PowerShell environment for an offline CPU repeat:

```powershell
$env:REQCLUSTER_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
$env:REQCLUSTER_EMBEDDING_DEVICE = 'cpu'
$env:REQCLUSTER_LLM_PROVIDER = 'mock'
$env:HF_HUB_OFFLINE = '1'
$env:OMP_NUM_THREADS = '1'
$env:NUMBA_NUM_THREADS = '1'
```

For Experiment A, load the workbook with `openpyxl`, encode the `Content` column once with `reqcluster.embeddings.get_embedder().encode(texts, 64)`, then for each seed 0–19 call `reqcluster.ml.reduce.reduce(embeddings, seed)` and `reqcluster.ml.cluster.hdbscan(x10, 5, 3)`. Compare the resulting labels with the workbook `Group` column using scikit-learn ARI and NMI, including `-1` as a predicted label. For Experiment B reuse seed 0's `x10` and call `hdbscan(x10, min_cluster_size, min_samples)` for the combinations in the table.

The workbook is a small, single-label reference; engineering requirements can legitimately overlap. The parameter sweep is exploratory and the reference labels were used to assess it; it is not a held-out test. No real-enrichment ablation, Postgres integration, GPU comparison, browser walkthrough or repeated-seed enriched evaluation was performed in these experiments.

## Priorities for further investigation

1. Reproduce the supplied enriched runs with saved enriched texts, embeddings, model version, device, full parameters and effective seed; compare base/enriched/hybrid using identical seeds and inputs.
2. Evaluate per-group correct-theme recall, noise rate, PID and output requirements, and cluster stability together. Avoid declaring success from coverage alone.
3. Apply the verified working-tree fixes for pre-clustering coverage and single-cluster Hungarian accuracy if they have not yet been committed.
4. Clarify the meaning of Ward's membership values and noise policy; do not present placeholder values as calibrated confidence.
5. Record complete provenance and use CSV rather than truncated PDFs for automated membership evaluation.
