# Phase 2 Enrichment API and Database Persistence

This document describes Phase 2 Task 3: connecting the Task 2 enrichment service to the backend API and database so Task 1 enriched and hybrid embeddings can be used publicly.

## Endpoint Overview

`POST /api/enrich`

Runs enrichment for an uploaded session and persists one enrichment row per requirement.

Request shape:

```text
session_id: positive integer
provider_name: mock | openai_compatible | local, default mock
embedding_mode: enriched | hybrid, default hybrid
batch_size: 1..64, default 8
max_concurrency: 1..16, default 4
timeout_seconds: 1..120, default 30
force_refresh: boolean, default false
fail_fast: boolean, default false
use_cache: boolean, default true
```

Response shape:

```text
session_id
status
total
succeeded
failed
provider
model
prompt_version
domain_vocabulary
quality_report
warnings
duration_ms
```

`GET /api/enrich/status/{session_id}`

Returns persisted enrichment readiness counts:

```text
session_id
status
total
succeeded
failed
pending
latest_run_created_at
provider
model
warnings
```

`GET /api/enrich/results?session_id=...`

Returns persisted enrichment rows in original requirement order:

```text
requirement_id
expanded_text
domain_terms
functional_intent
mentioned_components
assumptions
confidence
warnings
quality_report
status
```

Internal database IDs are not exposed by the enrichment result API.

## Database Model

`EnrichedRequirement` stores durable enrichment results per session and requirement.

Important fields:

- `session_id`
- `requirement_db_id`
- `requirement_id`
- `requirement_text_hash`
- `provider`
- `model`
- `prompt_version`
- `embedding_mode_recommended`
- `expanded_text`
- `domain_terms_json`
- `functional_intent`
- `mentioned_components_json`
- `assumptions_json`
- `confidence`
- `warnings_json`
- `quality_report_json`
- `status`
- `error_message`
- `created_at`
- `updated_at`

The table has indexes for session and requirement lookup. A uniqueness constraint prevents uncontrolled duplicates for the same session, requirement, provider, model, prompt version, and requirement text hash.

The app still uses the existing `create_all` startup path. No migration framework has been introduced.

## Enrichment Lifecycle

1. Upload stores cleaned requirements as before.
2. `POST /api/enrich` loads requirements in stable `Requirement.id` order.
3. The service extracts deterministic domain vocabulary for the session.
4. Existing valid rows are reused unless `force_refresh=true`.
5. Missing or stale rows are sent to Task 2 `enrich_requirements`.
6. Successful and failed per-requirement results are persisted transactionally.
7. `GET /api/enrich/status/{session_id}` reports readiness.
8. `GET /api/enrich/results?session_id=...` returns rows in requirement order.
9. `POST /api/cluster` can use `embedding_mode=enriched` or `embedding_mode=hybrid` once enrichment is complete.

## Force Refresh

Default behavior reuses valid persisted rows with the same:

- provider
- model
- prompt version
- requirement text hash

`force_refresh=true` bypasses those rows and asks the Task 2 service to regenerate. The row is updated or replaced under the same uniqueness key, so repeated runs do not create uncontrolled duplicates.

## Cache vs Database

Task 2 owns the file cache:

```text
backend/cache/llm_enrichment/llm_enrichment_<sha256>.json
```

The file cache speeds provider calls and is controlled by `use_cache`.

Task 3 owns database persistence. The database is the application-level store used by public API endpoints and clustering.

## Clustering With Enriched Or Hybrid Embeddings

Base clustering remains unchanged:

```text
POST /api/cluster
embedding_mode=base
```

Enriched or hybrid clustering now requires persisted enrichment:

```text
POST /api/enrich
POST /api/cluster with embedding_mode=enriched or embedding_mode=hybrid
```

Before calling `run_pipeline`, the API reconstructs `enriched_texts` in the same order as the original requirements and validates:

- same count
- matching requirement database IDs
- matching requirement text hashes
- current prompt version
- successful status
- non-empty expanded text
- parseable persisted JSON metadata

If enrichment is missing or misaligned, the API returns:

```text
Run /api/enrich for this session before clustering with enriched or hybrid embeddings.
```

The public API does not silently fall back to base embeddings for enriched or hybrid requests.

## Error Cases

Safe errors include:

- session not found
- no requirements found for session
- invalid enrichment provider
- provider not configured
- enrichment failed
- enrichment missing for hybrid or enriched clustering
- enrichment alignment mismatch
- invalid API parameter bounds

Stack traces, full LLM responses, API keys, and raw provider request payloads are not returned to clients.

## Security Considerations

- Provider names are allowlisted by the API schema.
- `batch_size`, `max_concurrency`, and `timeout_seconds` are bounded.
- API keys and provider secrets are never stored or returned.
- Full requirement text is not placed in enrichment error messages.
- LLM output remains untrusted and is persisted only after Task 2 parsing and validation.
- Persisted JSON fields are parsed defensively before result retrieval or clustering use.
- Database writes are committed transactionally and rolled back on persistence failure.
- User-selected cache paths are not accepted.
- No external queues, Redis, Celery, or new infrastructure were added.

## Handoff To Task 4

The frontend can later call:

```text
POST /api/enrich
GET /api/enrich/status/{session_id}
GET /api/enrich/results?session_id=...
POST /api/cluster with embedding_mode=hybrid
```

Task 4 can add UI controls for provider selection, enrichment progress, quality warnings, and enriched/hybrid clustering without changing the backend flow described here.
