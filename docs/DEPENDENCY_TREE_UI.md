# Dependency Tree UI

The Dependency Tree page visualizes the persisted dependency-tree response from
the backend. The dependency inference is heuristic: edges are based on explicit
requirement references, data-flow wording, precondition wording, and semantic
similarity. It is not a formal proof of system dependency.

## Large Graph Simplification

Large dependency graphs are simplified by default so the page remains usable.
When a tree has more than 250 nodes or more than 600 edges, the UI shows:

```text
Large graph simplified for readability. Use filters or focus mode to explore more.
```

Default large-graph caps:

- Node limit: 250
- Edge limit: 400
- Max depth: 8

Hard frontend safety caps:

- Nodes rendered: 1000
- Edges rendered: 3000

For larger future datasets, use group, search, depth, and focus filters before
raising the limits.

## Filters

Available controls:

- View mode: 3D graph or 2D layered view
- Group filter: show one cluster/group plus immediate connected context
- Search: match requirement ID, requirement text, or group label
- Max depth: limit visible dependency depth, with a full-depth toggle
- Edge type: data, sequential, hierarchical, reference, or semantic when present
- Node limit: cap visible nodes
- Edge limit: cap visible dependencies
- Minimum strength: hide edges below a weight threshold
- Reset filters: restore the graph defaults for the current dataset

Edges are sorted by weight descending, then by source and target ID for stable
deterministic rendering. Nodes are prioritized by search match, focus match,
selected group, degree, depth, and requirement ID.

## Focus Mode

Clicking a node selects it. If focus depth is set to 1-hop or 2-hop, clicking
also filters the graph to the selected node's neighborhood:

- 1-hop: selected node, parents, and children
- 2-hop: selected node, parents/children, and their immediate neighbors

Use Clear focus to return to the current non-focused filter view.

## Hover And Details

Plotly hover labels are intentionally short. They show only requirement ID,
group, depth, degree, and a click hint. Full requirement text and rationale are
not included in Plotly hover labels.

Full node details are rendered as plain text in the bounded right-side panel:

- Requirement ID
- Group label
- Depth
- Parent and child counts
- Incoming and outgoing dependency counts
- Dependency type counts
- Requirement text
- Group rationale
- Parent and child previews

Long text wraps inside the panel and uses Show more / Show less. The UI does
not render backend strings as HTML or Markdown.

## Manual Test Checklist

1. Open Dependency Tree.
2. Confirm the page loads without freezing.
3. Confirm the simplified-mode message appears for a 600-requirement tree.
4. Confirm the graph is less dense than the full 600-node / 1980-edge tree.
5. Hover a node.
6. Confirm the hover label is short and bounded.
7. Confirm full text appears in the side panel and wraps.
8. Click a node.
9. Confirm focus mode shows the selected node and neighbors.
10. Search for `command authentication`.
11. Confirm matching nodes are highlighted and shown with neighbors.
12. Select a group from the dropdown.
13. Confirm the graph reduces to that group and connected context.
14. Change max depth.
15. Confirm the graph updates.
16. Toggle data and sequential edge types.
17. Confirm edge visibility changes.
18. Increase edge limit.
19. Confirm more edges appear without horizontal overflow.
20. Reset filters.
21. Confirm the default simplified view returns.
22. Open Similarity Graph and Scatter Plot.
23. Confirm those pages still work.
