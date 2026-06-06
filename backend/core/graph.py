"""Requirement similarity graph.

Edges are built with an approximate k-nearest-neighbour search (hnswlib) over
**all** non-noise requirements - O(N log N) - so the graph is representative at
50k, not a dense O(N^2) cosine matrix truncated to the first 500 nodes. Falls
back to a bounded exact computation if hnswlib is unavailable, so it never breaks.

Node structure is unchanged (id, node_id, requirement_text, cluster_id, x, y,
is_noise) to keep the dashboard's hover/inspect behaviour intact.
"""

import numpy as np
from typing import Any, Dict, List, Optional

from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_K = 10           # neighbours per node for the ANN graph
MAX_EDGES = 4000         # cap stored edges (highest similarity retained)
EXACT_NODE_CAP = 500     # fallback only: bound the dense computation


def _knn_edges(
    embeddings: np.ndarray, labels: np.ndarray, threshold: float, k: int
) -> Optional[List[Dict[str, Any]]]:
    """Approximate kNN edges over non-noise nodes via hnswlib. None if unavailable."""
    try:
        import hnswlib
    except Exception:
        return None

    idx = [i for i in range(len(labels)) if labels[i] != -1]
    if len(idx) < 2:
        return []

    X = np.asarray(embeddings, dtype=np.float32)[idx]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    m, dim = Xn.shape

    try:
        index = hnswlib.Index(space="cosine", dim=dim)
        index.init_index(max_elements=m, ef_construction=200, M=16)
        index.add_items(Xn, np.arange(m))
        index.set_ef(max(k + 1, 64))
        kk = min(k + 1, m)
        neighbours, distances = index.knn_query(Xn, k=kk)
    except Exception:
        return None

    edges: List[Dict[str, Any]] = []
    seen = set()
    for li in range(m):
        gi = idx[li]
        for lj, dist in zip(neighbours[li], distances[li]):
            lj = int(lj)
            if lj == li:
                continue
            sim = 1.0 - float(dist)  # hnswlib cosine space: distance = 1 - cos
            if sim < threshold:
                continue
            gj = idx[lj]
            a, b = (gi, gj) if gi < gj else (gj, gi)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append({"source": a, "target": b, "weight": round(sim, 4)})
    return edges


def _exact_edges(
    embeddings: np.ndarray, labels: np.ndarray, threshold: float
) -> List[Dict[str, Any]]:
    """Bounded exact fallback (dense cosine over the first EXACT_NODE_CAP non-noise nodes)."""
    n = len(labels)
    if n > EXACT_NODE_CAP:
        compute_idx = [i for i in range(n) if labels[i] != -1][:EXACT_NODE_CAP]
    else:
        compute_idx = list(range(n))
    edges: List[Dict[str, Any]] = []
    if len(compute_idx) <= 1:
        return edges
    sim_matrix = cosine_similarity(embeddings[compute_idx])
    for i in range(len(compute_idx)):
        for j in range(i + 1, len(compute_idx)):
            sim = float(sim_matrix[i, j])
            if sim >= threshold:
                edges.append({
                    "source": compute_idx[i], "target": compute_idx[j], "weight": round(sim, 4),
                })
    return edges


def build_similarity_graph(
    embeddings: np.ndarray,
    texts: List[str],
    labels: np.ndarray,
    umap_2d: np.ndarray,
    req_ids: List[str],
    threshold: float = 0.65,
    k: int = DEFAULT_K,
    max_edges: int = MAX_EDGES,
) -> Dict[str, Any]:
    """Build the similarity graph (nodes + weighted edges)."""
    n = len(texts)

    nodes = [
        {
            "id": i,
            "node_id": req_ids[i] if i < len(req_ids) else f"REQ-{i}",
            "requirement_text": texts[i],
            "cluster_id": int(labels[i]),
            "x": float(umap_2d[i, 0]),
            "y": float(umap_2d[i, 1]),
            "is_noise": bool(labels[i] == -1),
        }
        for i in range(n)
    ]

    edges = _knn_edges(embeddings, labels, threshold, k)
    if edges is None:  # hnswlib unavailable -> safe exact fallback
        edges = _exact_edges(embeddings, labels, threshold)

    if len(edges) > max_edges:
        edges.sort(key=lambda e: e["weight"], reverse=True)
        edges = edges[:max_edges]

    return {"nodes": nodes, "edges": edges}
