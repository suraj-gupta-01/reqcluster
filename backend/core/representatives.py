"""Representative requirement extraction.

Extracts requirements closest to the cluster centroid as human-reviewable
cluster summaries. Uses cosine similarity in the original 384-dim embedding
space.

This module does not modify any database state or run LLM calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Representative:
    """A representative requirement for a cluster."""

    req_id: str
    text: str
    index: int
    similarity_to_centroid: float
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


def extract_representatives(
    embeddings: np.ndarray,
    texts: List[str],
    labels: np.ndarray,
    req_ids: List[str],
    top_n: int = 3,
) -> Dict[int, List[Representative]]:
    """Extract representative requirements for each cluster.

    For each cluster, returns the N requirements closest to the cluster
    centroid based on cosine similarity in the original embedding space.

    Args:
        embeddings: Original 384-dim embeddings, shape (N, D).
        texts: Requirement texts, length N.
        labels: HDBSCAN cluster labels, shape (N,).
        req_ids: Requirement IDs, length N.
        top_n: Number of representatives per cluster.

    Returns:
        Dict mapping cluster_id → list of Representative objects,
        sorted by similarity to centroid (highest first).
    """
    unique_clusters = sorted(set(int(l) for l in labels if l != -1))
    result: Dict[int, List[Representative]] = {}

    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        indices = np.where(mask)[0]
        cluster_emb = embeddings[indices]

        if cluster_emb.shape[0] == 0:
            result[cluster_id] = []
            continue

        # Compute centroid
        centroid = cluster_emb.mean(axis=0, keepdims=True)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm

        # Cosine similarity of each point to centroid
        similarities = cosine_similarity(cluster_emb, centroid).flatten()

        # Get top-N indices sorted by similarity (descending)
        n_reps = min(top_n, len(indices))
        top_local = np.argsort(similarities)[::-1][:n_reps]

        reps: List[Representative] = []
        for rank, local_idx in enumerate(top_local):
            global_idx = int(indices[local_idx])
            reps.append(
                Representative(
                    req_id=req_ids[global_idx] if global_idx < len(req_ids) else f"REQ-{global_idx}",
                    text=texts[global_idx] if global_idx < len(texts) else "",
                    index=global_idx,
                    similarity_to_centroid=round(float(similarities[local_idx]), 6),
                    rank=rank + 1,
                )
            )

        result[cluster_id] = reps

    return result


def extract_cluster_summary_texts(
    embeddings: np.ndarray,
    texts: List[str],
    labels: np.ndarray,
    req_ids: List[str],
    top_n: int = 3,
) -> Dict[int, str]:
    """Generate a summary text for each cluster from its representative requirements.

    Returns a dict mapping cluster_id → concatenated representative requirement texts,
    separated by semicolons. Suitable as a human-readable cluster summary.
    """
    reps_by_cluster = extract_representatives(embeddings, texts, labels, req_ids, top_n)
    summaries: Dict[int, str] = {}

    for cluster_id, reps in reps_by_cluster.items():
        if not reps:
            summaries[cluster_id] = ""
            continue
        texts_list = [rep.text.strip() for rep in reps if rep.text.strip()]
        summaries[cluster_id] = "; ".join(texts_list)

    return summaries
