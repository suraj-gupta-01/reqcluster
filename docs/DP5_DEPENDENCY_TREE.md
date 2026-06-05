# DP5: Dependency Tree + Rationale (Honeywell brief)

Satisfies the DP5 deliverables beyond clustering: a **dependency tree** and a
**rationale document**, with optional on-prem LLM narrative.

## What it does

For a clustered session, `core/dependency_tree.py` infers a directed acyclic
graph of relationships between requirements. Edge direction is
`source -> target` meaning *source is a prerequisite of target*.

### Relation types and signals

| Relation | Signal |
|----------|--------|
| `reference` / `hierarchical` | Explicit `REQ-xxx` mention in another requirement (e.g. "as defined in REQ-001"). Hierarchical when a compliance cue ("in accordance with", "as specified in") is present. |
| `data` | One requirement uses an output verb (provide/generate/output…) and a semantically similar one uses an input verb (use/require/consume…) over a shared salient term. |
| `sequential` | A requirement is gated by a precondition cue ("once", "after", "upon", "in response to") shared with a similar requirement. |
| `hierarchical` | Same cluster, one requirement materially more general than another at high similarity. |

Candidate pairs are gated by cosine similarity (top-k per node) to keep the
analysis meaningful and `O(n·k)`. Cycles are removed greedily by descending
edge weight to guarantee a DAG; tree **levels** are the longest path from a root.

The result is fully deterministic. Narrative prose (grouping rationale, edge
justifications surfaced in the UI) can be produced by the on-prem LLM provider.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/dependencies/generate` | Build + persist the tree and rationale document. Body: `{session_id, provider_name?, sim_threshold?, top_k?}`. |
| GET | `/api/dependencies?session_id=` | Retrieve nodes, edges, stats, and the rationale document. |

Response shape:

```json
{
  "session_id": 1,
  "nodes": [{"id": 0, "node_id": "REQ-001", "requirement_text": "...", "cluster_id": 0, "level": 0}],
  "edges": [{"source": 0, "target": 1, "relation": "reference", "weight": 0.95, "rationale": "..."}],
  "stats": {"n_nodes": 64, "n_edges": 120, "max_depth": 3, "relation_counts": {...}, "root_count": 8},
  "rationale": {"grouping": [...], "dependencies": [...]}
}
```

## UI

The **Dependency Tree** page provides two visualization modes:
- **2D View**: Lays nodes out vertically aligned by dependency level.
- **3D View**: Distributes nodes within each dependency level along a cylindrical Y-Z plane to prevent visual overlap and clutter. The radial coordinates are calculated as:
  $$R = 0.5 \times \sqrt{L}$$
  $$\theta_i = \frac{2 \pi i}{L}$$
  $$y_i = R \cos(\theta_i)$$
  $$z_i = R \sin(\theta_i)$$
  where $L$ is the number of requirements in that level, and $i$ is the node index. 
- Interactive camera controls allow panning, zooming, and rotating (orbit) in the 3D space. Edges are colored by relation type, and clicking nodes surfaces their detailed rationale.

## Notes

Dependency extraction from free text is heuristic. The deterministic core gives
explainable, reproducible edges; the LLM layer enriches the prose. For
ground-truth evaluation use the Dronology dataset (`data/README.md`).
