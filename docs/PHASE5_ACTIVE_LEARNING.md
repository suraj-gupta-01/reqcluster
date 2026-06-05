# Phase 5: Active Learning

Closes the human-in-the-loop loop. The must-link / cannot-link constraints
captured in Phase 4 (from manual cluster corrections) are injected back into the
clustering, and the least-certain requirements are surfaced for review.

## Constraint injection ("COP-HDBSCAN-lite")

HDBSCAN has no native constrained mode, so `core/constrained_clustering.py`
enforces constraints as a deterministic post-hoc repair over the current labels:

- **must-link** — connected components of must-linked points are collapsed onto
  a single representative cluster (union-find).
- **cannot-link** — when two cannot-linked points share a cluster, the
  lower-confidence point is moved to its nearest non-conflicting cluster, or to
  noise if none exists.

This is a repair layer, not a constrained optimiser: fast, deterministic, and it
never fails. It operates on the labels already stored for the session, so no
UMAP/HDBSCAN re-run is required.

## Uncertainty sampling

`core/active_learning.py` ranks requirements by clustering uncertainty (noise
points first, then low membership probability). The queue is the most valuable
set to send for human review.

## Quality tracking

Each constrained iteration records silhouette, noise rate, and cluster count to
the `clustering_iterations` table so improvement can be tracked across rounds.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/cluster/constrained` | Apply active constraints; records an iteration. Body: `{session_id}`. |
| GET | `/api/active-learning/queue?session_id=&top_k=` | Uncertainty-sampled review queue. |
| GET | `/api/quality/history?session_id=` | Quality metrics across iterations. |

## UI

The **Active Learning** page shows the uncertainty queue and a dual-axis
quality-history chart, with a one-click "Apply constraints" action.

## Workflow

1. Cluster a session.
2. On **Review Queue**, approve manual corrections (Phase 4). These generate
   must/cannot-link constraints.
3. On **Active Learning**, click **Apply constraints** to re-cluster under those
   constraints and watch silhouette improve / noise drop across iterations.
