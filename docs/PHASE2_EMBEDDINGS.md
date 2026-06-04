# Phase 2 Domain-Aware Embeddings

This document describes the Phase 2, Part 1 embedding integration for ReqCluster. It extends the Phase 1 SBERT pipeline without requiring an LLM service to exist yet.

## Embedding Modes

ReqCluster now supports three internal embedding modes:

| Mode | Input to SBERT | Behavior |
| --- | --- | --- |
| `base` | Original cleaned requirement text | Preserves Phase 1 behavior. |
| `enriched` | Enriched requirement text | Uses enriched text when available. If configured with `fallback_to_base=True`, missing enriched text falls back to the original requirement text. |
| `hybrid` | Original text plus enriched context | Uses deterministic section markers to combine both inputs. Missing enriched text falls back to original text only when fallback is explicitly enabled. |

Hybrid mode always uses this deterministic inert-text format:

```text
Original Requirement:
{original_text}

Domain-Aware Context:
{enriched_text}
```

The enriched text is treated only as plain text input to the embedding model. It is never executed, parsed as instructions, used as a file path, or logged by default.

## Cache Key Design

Domain-aware embeddings use SHA-256 cache keys and safe filenames:

```text
domain_embeddings_{sha256_hex}.npy
```

The hash payload includes:

- embedding model name
- embedding mode
- normalized ordered model input texts
- normalization flag
- fallback flag
- maximum text length
- cache namespace/version

`batch_size` is intentionally not part of the hash because it does not affect the resulting embedding values. Cache files are written to a temporary file first and then atomically replaced. Cache reads are ignored and regenerated when the array shape, dtype, finiteness, or L2 normalization checks fail.

Phase 1's default base pipeline remains available through the original `generate_embeddings` path so existing clustering behavior stays compatible.

## Comparison Metrics

`backend/core/embedding_comparison.py` compares base embeddings with enriched or hybrid embeddings and returns a JSON-serializable report:

- per-requirement cosine similarity
- per-requirement cosine distance delta: `1 - cosine_similarity`
- aggregate mean, median, min, and max cosine similarity
- aggregate mean delta
- count of requirements above delta thresholds `0.05`, `0.10`, and `0.20`
- top-k nearest-neighbor preservation using average Jaccard overlap
- optional cluster-impact metrics when both label sets are available:
  - adjusted Rand index
  - normalized mutual information
  - silhouette score delta when valid

Metric failures are returned as warnings instead of crashing the pipeline.

## Ablation Report

`backend/core/ablation.py` provides a read-only `run_embedding_ablation(...)` helper. It runs:

1. base embeddings
2. selected enriched or hybrid embeddings
3. UMAP 10D and 2D reductions for each, when valid
4. HDBSCAN clustering for each, when valid
5. c-TF-IDF labeling, when clusters exist
6. similarity graph construction, when 2D coordinates exist
7. embedding and cluster comparison metrics

The report shape is:

```json
{
  "mode": "hybrid",
  "n_requirements": 118,
  "base": {
    "embedding_shape": [118, 384],
    "n_clusters": 6,
    "noise_count": 0,
    "noise_rate": 0.0,
    "silhouette_score_10d": 0.41,
    "duration_ms": {}
  },
  "enriched": {
    "embedding_shape": [118, 384],
    "n_clusters": 6,
    "noise_count": 1,
    "noise_rate": 0.008475,
    "silhouette_score_10d": 0.44,
    "duration_ms": {}
  },
  "comparison": {},
  "warnings": []
}
```

The ablation runner does not import or mutate database state.

## API Status

The public `/api/cluster` request schema accepts:

```json
{
  "embedding_mode": "base",
  "enable_embedding_comparison": false,
  "run_ablation": false
}
```

Public API requests for `embedding_mode="base"` preserve the Phase 1 behavior.

Public API requests for `embedding_mode="enriched"` or `embedding_mode="hybrid"` now require persisted enrichment from:

```text
POST /api/enrich
```

If enrichment is missing or misaligned, `/api/cluster` returns HTTP 400 with a clear instruction to run `/api/enrich` first. The public API does not silently fall back to base embeddings for enriched or hybrid requests.

Internal callers can still pass `enriched_texts` directly into `run_pipeline(...)` or `run_embedding_ablation(...)`.

## LLM Enrichment Handoff

The enrichment service passes enriched requirement text as an ordered list aligned with the original requirements:

```python
run_pipeline(
    texts=cleaned_requirement_texts,
    req_ids=requirement_ids,
    embedding_mode="hybrid",
    enriched_texts=enriched_requirement_texts,
)
```

Ordering is part of the cache and API contract. The enriched list must have the same length and order as `texts`. Internal callers may pass `None` values and rely on `DomainEmbeddingConfig.fallback_to_base`; the public API validates persisted enrichment and rejects missing entries before calling the pipeline.

## Security Considerations

- No unsafe deserialization is used; NumPy cache reads use `allow_pickle=False`.
- No `eval`, `exec`, dynamic imports from user input, shell calls, or untrusted code execution are used.
- Cache filenames are derived only from safe SHA-256 hex digests.
- Full requirement text is not logged by the embedding modules.
- API parameters are bounded through Pydantic validation.
- Enriched text is untrusted inert text and cannot alter program behavior.
- Nearest-neighbor preservation caps pairwise comparison size to avoid large `N x N` memory blowups.
- Similarity graph edge caps from Phase 1 remain enforced.
