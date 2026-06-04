# Phase 2 Frontend Enrichment UI

This document describes Phase 2 Task 4: the React dashboard workflow that connects upload, enrichment, enriched/hybrid clustering, comparison metrics, and existing visualizations.

## UI Flow

The Phase 2 dashboard flow is:

```text
Upload requirements
Run enrichment
Inspect enriched text and domain vocabulary
Run base, enriched, or hybrid clustering
Inspect comparison and ablation metrics
Use Scatter, Graph, Requirements, Cluster Detail, and Overview pages as before
```

Phase 1 upload and base clustering remain available.

## New Route

```text
/enrichment
```

The sidebar now includes an `Enrichment` item. The page works even when the active session is not in the URL because it loads uploaded sessions from `/api/sessions`.

## API Endpoints Used

The frontend uses centralized helpers from `frontend/src/utils/api.js`:

```text
POST /api/enrich
GET /api/enrich/status/{session_id}
GET /api/enrich/results?session_id=...
POST /api/cluster
GET /api/progress/{session_id}
GET /api/sessions
GET /api/sessions/{session_id}
GET /api/requirements?session_id=...
```

Backend errors are normalized before display. API keys, provider secrets, and stack traces are not shown.

## Provider Controls

The Enrichment page exposes:

```text
provider_name: mock | openai_compatible | local
embedding_mode: hybrid | enriched
batch_size: 1..64
max_concurrency: 1..16
timeout_seconds: 1..120
force_refresh
use_cache
fail_fast
```

The default provider is `mock`, which is offline and safe for testing.

## Status Behavior

`EnrichmentStatusCard` displays:

- status
- total
- succeeded
- failed
- pending
- provider
- model
- latest run time
- warnings

After `POST /api/enrich`, the UI polls `GET /api/enrich/status/{session_id}` every second and stops when enrichment is complete, failed, or the polling cap is reached. The interval is cleaned up on unmount.

## Results Table

`EnrichmentResultsTable` displays persisted enrichment rows in requirement order.

Collapsed rows show:

- requirement ID
- expanded text preview
- confidence
- warning count
- status

Expanded rows show:

- original requirement text
- full expanded text
- functional intent
- domain terms
- mentioned components
- assumptions
- warnings
- quality metrics

The table includes search, status filtering, low-confidence filtering, warning filtering, and 25-row pagination.

## Vocabulary Tags

`DomainVocabularyTags` displays session-level vocabulary from `POST /api/enrich`. If a page is loaded after enrichment and the response is not in memory, the component falls back to domain terms from persisted result rows.

Empty state:

```text
No domain vocabulary available yet.
```

## Clustering Controls

The Enrichment page can run:

- base clustering
- enriched clustering
- hybrid clustering

Base clustering is always allowed for uploaded sessions and preserves Phase 1 behavior.

Enriched and hybrid clustering require complete persisted enrichment. If enrichment is missing, the UI shows:

```text
Run enrichment for this session before clustering with enriched or hybrid embeddings.
```

Cluster options:

```text
similarity_threshold
min_cluster_size
min_samples
enable_embedding_comparison
run_ablation
```

After clustering succeeds, the page shows cluster count, noise count, embedding mode, and links to Scatter and Graph.

## Comparison Panel

`EmbeddingComparisonPanel` displays Task 1 comparison metrics when returned by `/api/cluster`:

- mean cosine similarity
- median cosine similarity
- min cosine similarity
- max cosine similarity
- mean delta
- delta threshold counts
- nearest-neighbor preservation
- warnings

Empty state:

```text
Embedding comparison will appear after clustering with comparison enabled.
```

## Ablation Panel

`AblationReportPanel` displays Task 1 ablation summaries when returned by `/api/cluster`:

- base embedding shape, cluster count, noise count, noise rate, silhouette
- enriched/hybrid embedding shape, cluster count, noise count, noise rate, silhouette
- cluster delta
- noise-rate delta
- silhouette delta
- warnings

Empty state:

```text
Ablation report will appear after clustering with ablation enabled.
```

## Scatter Integration

The Scatter page remains compatible with Phase 1 data and still renders the latest persisted clustering result.

It now shows:

- latest known embedding mode from the Enrichment page clustering action
- a view selector for base, enriched, hybrid, or latest
- a clear fallback message when true before/after visualization is unavailable

Current limitation:

```text
The backend persists only the latest scatter coordinates for a session.
```

True side-by-side before/after scatter comparison requires backend support for storing multiple clustering runs. The UI does not invent comparison coordinates.

## Requirements And Overview Integration

The Overview page shows a compact Phase 2 enrichment summary card.

The Requirements page shows:

- enriched available badge
- confidence badge
- warning count badge
- expandable enriched text and domain tags

These indicators are loaded opportunistically and do not block Phase 1 requirement browsing.

## Security Considerations

- No `dangerouslySetInnerHTML`.
- LLM text and warnings are rendered as plain text.
- Provider secrets are not entered or stored in the frontend.
- Backend errors are normalized and redacted.
- No Markdown or HTML rendering is added.
- API URLs use the existing Axios base URL and normal query parameters.
- The UI does not log full requirement text.

## Manual Test Checklist

1. Start backend.
2. Start frontend.
3. Upload sample requirements.
4. Run base clustering from Upload or Enrichment.
5. Open Scatter and Graph; confirm Phase 1 still works.
6. Open Enrichment.
7. Run enrichment with `mock`.
8. Confirm status reaches complete.
9. Confirm results and domain vocabulary display.
10. Run hybrid clustering with comparison enabled.
11. Confirm comparison metrics display when returned.
12. Run hybrid clustering with ablation enabled.
13. Confirm ablation summary displays when returned.
14. Open Scatter after hybrid clustering.
15. Refresh the browser and confirm the app reloads.
16. Try enriched/hybrid clustering before enrichment on a fresh session and confirm the clear error.

## Known Limitations

- Frontend tests are not configured in this project. Verification currently uses production build, backend tests, and manual checks.
- The backend stores the latest clustering graph/scatter state only, so true persisted side-by-side before/after visualization is a future backend enhancement.
