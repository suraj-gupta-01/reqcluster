import numpy as np
from typing import List, Dict, Any
from sklearn.metrics.pairwise import cosine_similarity


def build_similarity_graph(
    embeddings: np.ndarray,
    texts: List[str],
    labels: np.ndarray,
    umap_2d: np.ndarray,
    req_ids: List[str],
    threshold: float = 0.65,
) -> Dict[str, Any]:
    """
    Build a similarity graph for visualization.
    
    Nodes: requirements
    Edges: pairs with cosine similarity > threshold
    
    Returns dict with 'nodes' and 'edges' lists.
    """
    n = len(texts)

    # Build nodes
    nodes = []
    for i in range(n):
        nodes.append({
            "id": i,
            "node_id": req_ids[i] if i < len(req_ids) else f"REQ-{i}",
            "requirement_text": texts[i],
            "cluster_id": int(labels[i]),
            "x": float(umap_2d[i, 0]),
            "y": float(umap_2d[i, 1]),
            "is_noise": bool(labels[i] == -1),
        })

    # Build edges using cosine similarity on original embeddings
    # For large N, sample to keep graph manageable
    max_nodes_for_full_graph = 500
    if n > max_nodes_for_full_graph:
        # Only compute edges for non-noise nodes
        non_noise_idx = [i for i in range(n) if labels[i] != -1]
        compute_idx = non_noise_idx[:max_nodes_for_full_graph]
    else:
        compute_idx = list(range(n))

    edges = []
    if len(compute_idx) > 1:
        sub_embeddings = embeddings[compute_idx]
        sim_matrix = cosine_similarity(sub_embeddings)

        for i in range(len(compute_idx)):
            for j in range(i + 1, len(compute_idx)):
                sim = float(sim_matrix[i, j])
                if sim >= threshold:
                    edges.append({
                        "source": compute_idx[i],
                        "target": compute_idx[j],
                        "weight": round(sim, 4),
                    })

    # Limit edges to top 2000 by weight to keep graph responsive
    if len(edges) > 2000:
        edges.sort(key=lambda e: e["weight"], reverse=True)
        edges = edges[:2000]

    return {"nodes": nodes, "edges": edges}
