# Amendment 6: second review (architect, 2026-11-19) and fix plan

Ids are referenced by commits (`R2-M1` etc.). Experiments ran on `var/cache/gate` embeddings, seed 42.

## Must fix
- **M1** After a failed/cancelled re-run (or restart sweep) the session status is `error`, and `frontend/src/components/SessionGate.jsx:28` renders only when `done`, so Overview/Scatter/Graph/ClusterDetail hide a valid revision. Fix (frontend): render when `status==='done' || revision>0`; on `error` show banner "Last run {last_job.status}; showing revision N". API test locks: cancel a re-run -> `revision>=1`, `status=="error"`.
- **M2** `llm/enrich.py:414-420` marks fallback results `succeeded`; summary and `texts_for_mode` trust it, so a misconfigured provider silently feeds template text into enriched clustering. Fix: `source=="fallback" and provider!="mock"` -> `status="failed"`, `error=res.error`. Test: provider openai without base url -> all failed, enriched cluster start 409.
- **M3** Probabilities go stale after edits: `services/assignments.py` updates only cluster_id; `active_learning.py:41-42` reads npz probabilities; `feedback.py:236,250` copies npz probabilities into the constrained run. Fix: reassign sets `probability=1.0`; uncertainty reads AssignmentRow probabilities; exclude requirements with non-rejected feedback in this run from the queue; recluster passes DB probabilities to persist_run. Test: move noise into cluster c -> not in /uncertainty, requirement probability 1.0.
- **M4** `services/feedback.py:158-172` review lacks `_check_idle` (A2) and a pending check. Cases: reject during a recluster is lost; reject then approve leaves "approved" with move reverted. Fix: `_check_idle(sid)` first; `Conflict` if `fb.status != "pending"`. Tests for both.
- **M5** `ml/refine.py` split/merge suggestions mostly false: on gold uav2k groups GMM-BIC flags 16/17 pure groups, none has silhouette_delta>0; multidomain 14/24 flagged (7 positive); merges 6 suggested, 1 positive; spread gate (0.3) never fires (gold spread max .149). Fix: skip splits/merges with `None` or `silhouette_delta <= 0`; sort splits by silhouette_delta; spread informational only. Test: forced-bimodal random sub_labels on a compact cluster -> no split.
- **M6** Coherence thresholds .8/.6/.4 (`ml/refine.py:83-90`) wrong for MiniLM: gold uav2k median .48 ("Low"), multidomain median .39 (54% "Very low"), random .20-.24. Fix: recalibrate (proposal >=.45 high, >=.35 moderate, >=.28 low) after measuring gold vs merged-pairs vs random groups.
- **M7** Enriched/hybrid runs: clustering embeds mode texts, but refinement (`refinement.py:95,217`), dependencies (`:78`) and constraint repair (`feedback.py:242`) embed base text (wrong space; first /coherence embeds 10k inside a request). Fix: one helper `run_embeddings(sid, run)` in services/clustering.py using `run.params["embedding_mode"]` (fallback to base + warning on Conflict); use at all 4 sites. Test: suggest job embeds enriched texts.

## Should fix
- **S1** Split id `max(ClusterRow)+1` reuses ids removed by a merge (merge (0,5) then split -> new 5; old feedback/audit alias). Fix: `RunRow.max_cluster_id` set at persist, bumped on split; reject-revert into a merged-away id also. Test.
- **S2** Dependency tree keyed on run only; stale after moves/merges/splits (node cluster ids, same-cluster edges, exports). Fix: store edit_seq in tree params, return `stale` in TreeOut; page shows rebuild notice.
- **S3** Review queue loads 1000 requirements to look up text/ids. Fix: `FeedbackOut.text`, conflict entries carry req_ids; delete the frontend fetch.
- **S4** `ml/graph.py` keeps global top 4000 edges: on uav10k all >= .940 and 4809/9998 nodes isolated; threshold slider has no effect. Fix: per-node best 2 above threshold first, then fill budget by weight. Test.
- **S5** GraphPage uses SVG `scatter` for up to 50k nodes -> `scattergl`.
- **S6** Scatter and Enrichment pages page through all requirements (50 requests at 50k). Fix: GraphNode gains `probability`, `attributes`; Scatter uses getGraph; EnrichmentOut gains `text`; remove loops.
- **S7** No compression. `GZipMiddleware(minimum_size=1024)` + nginx gzip.
- **S8** Ablation reruns the main pipeline (extra UMAP). Fix: reuse the main result for `params.embedding_mode` (2 pipeline runs, not 3).
- **S9** Per-cluster/per-suggestion LLM narratives run sequentially on the single CPU job worker. Fix: `cluster_narratives(...)` in narrate.py via `gateway.map`.
- **S10** Dependency tree payload has every requirement as a node with full text. Fix: only nodes in an edge; keep `stats.n_nodes`.
- **S11** Uncertainty: centroid margin beats `1-prob` (AUROC uav2k .932 vs .882, multidomain .778 vs .610, adversarial .785 vs .670); noise ties fall back to upload order. Fix: rank by negative cosine margin (own centroid minus best other); noise by gap between two nearest centroids. Depends on M3, M7.
- **S12** LLM cache = one JSON file per prompt (10k files). Fix: SQLite key-value like store.py.
- **S13** Docker: multi-stage (runtime copies .venv once), prune tests dirs (~70 MB) and torch/include (62 MB), drop /bin/uv from runtime, run `/app/.venv/bin/reqcluster-api`; keep .pyc.
- **S14** pandas (80 MB) only used in ingest.py: replace with csv + openpyxl read_only; pandas -> eval extra.
- **S15** ActiveLearningPage text wrong (moves apply immediately; recluster uses pending constraints). Fix text; return `moved` count from recluster.

## Nice to have
N1 Overview bars scale to 40 -> use max cluster size. N2 Requirements default sort `req_id` (string) -> `ordinal`. N3 reassign updates counts but not `run.metrics` (two n_clusters values). N4 npz unlink suppresses only FileNotFoundError (Windows PermissionError -> 500): suppress OSError, `with np.load`. N5 list_sessions N+1 job query. N6 Overview downloads whole /enrichment to count. N7 dead code: `clustering._scaled`, `refine.summary_texts`, `ml/active.quality` (tests only), `embeddings` build_texts re-export, `analysis.build_embedding_texts`, frontend assets hero.png/react.svg/vite.svg, private `_guard` import (make public). N8 plotly full bundle 4.4 MB (measure a partial bundle). N9 uncertainty loads x10 only for a length check; enriched pre-check loads rows twice.

## Checked, no change
Stopwords in keywords (same gold hit@5), embedding cache lookups (35k keys 0.08 s), per-edit relabel (0.08 s at 10k).

## Measurements to run after fixes
Payload size/time with and without gzip at 35k; scatter vs scattergl render; graph coverage after S4; suggestion precision (ARI delta per applied suggestion) before/after M5; coherence calibration (M6); uncertainty oracle loop (S11); image size/cold start after S13/S14; ablation wall time after S8. Optional: ONNX Runtime MiniLM to drop torch (~1 GB) only if it passes the ARI gate and small-N stability.
