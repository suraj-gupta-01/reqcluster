# Amendment 4: ML defaults (from findings/experiments.md)

Overrides earlier amendments where they conflict.

## Production default pipeline
- Embedder: `all-MiniLM-L6-v2`, torch backend, device auto (cuda > cpu). Within ~0.03 ARI of the best model and >=10x faster.
- Reduce: PCA(50, randomized, seed) -> UMAP(n_components=10, metric="cosine", n_neighbors=min(80, max(15, n // 100)), min_dist=0.0, n_epochs=200, random_state=seed). Use `init="random"` when n < 20; keep the default (spectral) init otherwise unless tests show instability.
- Clusterer: HDBSCAN(min_cluster_size=max(5, n // 50), min_samples=3, cluster_selection_method="eom").
- 2D view: UMAP(2) fitted on the 10D output (amendment 1).
- Numba JIT cache: set `NUMBA_CACHE_DIR={home}/cache/numba` before importing umap. This saves ~35 s on cold start.
- Measured at 10k rows on a quiet CPU: ~78 s total (encode 54 s, UMAP 23 s, HDBSCAN 1.2 s), versus ~150 s for legacy. ARI 0.911 (legacy 0.904).

## Optional modes (config, not default)
- `clusterer="ward"`: agglomerative ward on the 10D output, k chosen by sampled silhouette over [8, min(40, n // 5)]. Produces no noise. Best on UAV-10k (ARI 0.974), worse on the multidomain set.
- `embedding_model="nomic-ai/nomic-embed-text-v1.5"`: "accuracy" mode, +0.02-0.03 ARI on the hard sets, ~10x slower. Needs `trust_remote_code=True`, `torch_dtype=float32`, the `clustering: ` prefix, and the `einops` package (put it in an `accuracy` extra).

## Dropped
- ONNX backend: no speedup (fp32 169 vs torch 185 texts/s), and int8 costs accuracy. Remove `embedding_backend="onnx"` and the `onnx` extra from amendment 1. The Embedder Protocol keeps only torch and mock.
- Embedder candidates rejected: bge-s, bge-b, gte-b, e5-b, mpnet, mxbai, qwen3-0.6B (slower, and not more accurate).
- HDBSCAN on PCA or raw embeddings: collapses on multidomain (ARI ~0.1). UMAP stays required for the clustering space.

## Eval gates (eval/datasets.yaml baselines to set after S29)
| dataset | metric | floor |
|---|---|---|
| uav_fms_2k | ARI | 0.92 |
| uav_fms_10k | ARI | 0.89 |
| multidomain | ARI | 0.62 |
| adversarial | ARI | 0.32 |

## Known limits
- No configuration detects the adversarial noise rows well (~8% predicted noise versus 20% true). This is a future work item, not a blocker.
- PROMISE gives ARI ~0.05 for every configuration, so it is uninformative as a clustering benchmark.
