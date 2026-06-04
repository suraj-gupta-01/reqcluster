# Phase 2 LLM Requirement Enrichment

This document describes Phase 2 Task 2 for ReqCluster: a safe enrichment layer that prepares meaning-preserving requirement expansions before the existing Phase 2 domain-aware embedding pipeline runs.

## Purpose

The enrichment layer takes cleaned requirement text from Phase 1 preprocessing and returns:

- ordered enriched requirement text
- deterministic domain vocabulary
- per-requirement quality metadata
- structured provider errors and warnings

It does not run UMAP, HDBSCAN, c-TF-IDF labeling, graph construction, API persistence, or database writes.

The intended flow is:

```python
handoff = await enrich_then_prepare_pipeline_inputs(
    requirement_texts=cleaned_requirement_texts,
    requirement_ids=requirement_ids,
    provider_name="mock",
)

run_pipeline(
    texts=handoff["original_texts"],
    req_ids=handoff["requirement_ids"],
    embedding_mode="hybrid",
    enriched_texts=handoff["enriched_texts"],
)
```

## Providers

The provider abstraction lives in `backend/llm_services/providers.py`.

Supported providers:

- `MockLLMProvider`: offline, deterministic, no network, no API key. It uses only the original text and deterministic vocabulary extraction.
- `OpenAICompatibleProvider`: calls an OpenAI-compatible `/chat/completions` endpoint.
- `LocalLLMProvider`: calls a configured local HTTP endpoint.

Provider selection is passed to `enrich_requirements(..., provider_name=...)`. The default is `mock`.

## Configuration

OpenAI-compatible provider environment variables:

- `REQCLUSTER_LLM_PROVIDER`
- `REQCLUSTER_LLM_BASE_URL`
- `REQCLUSTER_LLM_API_KEY`
- `REQCLUSTER_LLM_MODEL`
- `REQCLUSTER_LLM_TIMEOUT_SECONDS`
- `REQCLUSTER_LLM_MAX_RETRIES`

Local provider environment variables:

- `REQCLUSTER_LOCAL_LLM_URL`
- `REQCLUSTER_LOCAL_LLM_MODEL`
- `REQCLUSTER_LOCAL_LLM_TIMEOUT_SECONDS`

The local provider only accepts `http` and `https` URLs. File URLs and unsafe schemes are rejected.

## Prompt Design

`backend/llm_services/prompts.py` defines `PROMPT_VERSION` and `build_requirement_expansion_prompt(...)`.

The prompt requires strict JSON only and instructs the model to:

- preserve the original requirement meaning
- avoid new obligations or stronger obligation language
- avoid invented numbers, standards, subsystems, interfaces, and verification criteria
- identify functional intent
- identify explicitly mentioned components
- identify useful domain vocabulary
- return no Markdown and no code fences

## Response Schema

Provider responses are parsed as untrusted JSON with this logical shape:

```json
{
  "expanded_text": "string",
  "domain_terms": ["string"],
  "functional_intent": "string",
  "mentioned_components": ["string"],
  "assumptions": ["string"],
  "confidence": 0.0,
  "warnings": ["string"]
}
```

Batch results use:

- `RequirementExpansionResult`
- `EnrichmentBatchResult`

Both expose `to_dict()` for JSON-safe serialization. Secrets are never included.

## Vocabulary Extraction

`backend/llm_services/vocabulary.py` provides deterministic vocabulary extraction without an LLM.

It extracts unigram, bigram, and trigram candidates with:

- lowercase normalization
- punctuation cleanup
- requirement stopword removal
- term frequency
- document frequency
- TF-IDF-like scoring
- stable sorting

The default output is the top 50 domain terms.

## Quality Metrics

`backend/llm_services/quality.py` compares original and enriched text.

Checks include:

- length ratio
- lexical overlap
- domain term coverage
- missing original critical tokens
- invented numeric values
- strengthened obligation language
- hallucination risk score
- adjusted confidence score
- per-requirement warnings

The quality layer warns conservatively. It rejects only empty, unsafe, or invalid provider outputs through the parser/service path.

## Cache Design

The enrichment cache is separate from embedding caches.

Default folder:

```text
backend/cache/llm_enrichment/
```

Filename format:

```text
llm_enrichment_<sha256>.json
```

The cache key includes:

- prompt version
- provider name
- model name
- normalized original requirement text
- domain vocabulary
- enrichment config version

Cache files are JSON only, written atomically, and read defensively. Corrupted cache files are ignored and regenerated. Requirement text is never used as a file path.

## Security Considerations

- No `eval`, `exec`, shell execution, dynamic imports from user input, YAML parsing, or unsafe deserialization.
- LLM output is parsed with `json.loads` only.
- Markdown fences and top-level arrays are rejected.
- Input and output sizes are bounded.
- API keys are never logged or returned.
- Provider HTTP calls use timeouts.
- OpenAI-compatible retries are bounded.
- Cache filenames are SHA-256 digests only.
- Local provider URLs must use `http` or `https`.
- Returned content is treated only as inert text.

## Handoff To Task 1 Embeddings

Task 1 already added `base`, `enriched`, and `hybrid` embedding modes. This enrichment task feeds that pipeline by returning an ordered `enriched_texts` list.

Use:

```python
enriched_texts = extract_enriched_texts(batch_result)

run_pipeline(
    texts=cleaned_requirement_texts,
    req_ids=requirement_ids,
    embedding_mode="hybrid",
    enriched_texts=enriched_texts,
)
```

Missing enrichment entries are returned as `None`, allowing the existing hybrid embedding fallback behavior to handle partial failures.

## Future Task 3 API/Database Handoff

The current public `/api/cluster` route still rejects `enriched` and `hybrid` modes because enriched text is not persisted yet. A future API/database task can store `RequirementExpansionResult` values, expose enrichment endpoints, and pass stored enriched text into `run_pipeline(...)`.

Until then, the enrichment service remains a read-only preparation layer.
