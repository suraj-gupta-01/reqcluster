"""Cluster merge candidate detection via silhouette analysis.

This module identifies clusters that may benefit from merging based on:
- Centroid cosine similarity (high similarity → merge candidate)
- Silhouette score delta (positive delta → merge improves coherence)

It does not modify any database state or run LLM calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class MergeCandidate:
    """A pair of clusters with high centroid similarity."""

    cluster_a: int
    cluster_b: int
    centroid_similarity: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MergeScore:
    """Silhouette score delta for a hypothetical merge."""

    cluster_a: int
    cluster_b: int
    silhouette_before: float
    silhouette_after: float
    silhouette_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MergeSuggestion:
    """A ranked merge suggestion with scores and metadata."""

    cluster_a: int
    cluster_b: int
    cluster_a_size: int
    cluster_b_size: int
    centroid_similarity: float
    silhouette_delta: float
    coherence_a: float
    coherence_b: float
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_cluster_centroids(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[int, np.ndarray]:
    """Compute mean embedding vector per cluster in the original embedding space.

    Noise points (label -1) are excluded.

    Returns:
        Dict mapping cluster_id → centroid vector (1-D array).
    """
    centroids: Dict[int, np.ndarray] = {}
    unique_labels = sorted(set(int(l) for l in labels if l != -1))
    for cluster_id in unique_labels:
        mask = labels == cluster_id
        cluster_embeddings = embeddings[mask]
        if cluster_embeddings.shape[0] == 0:
            continue
        centroid = cluster_embeddings.mean(axis=0)
        # L2-normalize the centroid for consistent cosine similarity
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids[cluster_id] = centroid
    return centroids


def compute_pairwise_cluster_similarity(
    centroids: Dict[int, np.ndarray],
) -> List[MergeCandidate]:
    """Compute cosine similarity between all cluster centroid pairs.

    Returns candidates sorted by similarity (highest first).
    """
    cluster_ids = sorted(centroids.keys())
    if len(cluster_ids) < 2:
        return []

    centroid_matrix = np.array([centroids[cid] for cid in cluster_ids])
    sim_matrix = cosine_similarity(centroid_matrix)

    candidates: List[MergeCandidate] = []
    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            candidates.append(
                MergeCandidate(
                    cluster_a=cluster_ids[i],
                    cluster_b=cluster_ids[j],
                    centroid_similarity=float(sim_matrix[i, j]),
                )
            )

    candidates.sort(key=lambda c: c.centroid_similarity, reverse=True)
    return candidates


def _compute_intra_cluster_coherence(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
) -> float:
    """Average pairwise cosine similarity within a cluster."""
    mask = labels == cluster_id
    cluster_emb = embeddings[mask]
    n = cluster_emb.shape[0]
    if n < 2:
        return 1.0
    sim = cosine_similarity(cluster_emb)
    # Average of upper triangle (exclude diagonal)
    upper_sum = float(np.triu(sim, k=1).sum())
    n_pairs = n * (n - 1) / 2
    return upper_sum / n_pairs if n_pairs > 0 else 1.0


def evaluate_merge_silhouette(
    embeddings_10d: np.ndarray,
    labels: np.ndarray,
    cluster_a: int,
    cluster_b: int,
) -> Optional[MergeScore]:
    """Compute silhouette score delta if two clusters were merged.

    A positive delta means the merge improves overall clustering coherence.

    Returns None if silhouette cannot be computed (e.g., < 2 clusters remain).
    """
    unique_clusters = set(int(l) for l in labels if l != -1)
    if cluster_a not in unique_clusters or cluster_b not in unique_clusters:
        return None
    if len(unique_clusters) < 3:
        # After merge we'd have < 2 clusters; silhouette undefined.
        return None

    # Only compute on non-noise points
    non_noise_mask = labels != -1
    filtered_embeddings = embeddings_10d[non_noise_mask]
    filtered_labels = labels[non_noise_mask].copy()

    if filtered_embeddings.shape[0] < 3:
        return None

    try:
        score_before = silhouette_score(filtered_embeddings, filtered_labels)
    except ValueError:
        return None

    # Create hypothetical merged labels
    merged_labels = filtered_labels.copy()
    merged_labels[merged_labels == cluster_b] = cluster_a

    # Check we still have >= 2 unique labels
    if len(set(merged_labels)) < 2:
        return None

    try:
        score_after = silhouette_score(filtered_embeddings, merged_labels)
    except ValueError:
        return None

    return MergeScore(
        cluster_a=cluster_a,
        cluster_b=cluster_b,
        silhouette_before=round(float(score_before), 6),
        silhouette_after=round(float(score_after), 6),
        silhouette_delta=round(float(score_after - score_before), 6),
    )


def suggest_merges(
    embeddings: np.ndarray,
    embeddings_10d: np.ndarray,
    labels: np.ndarray,
    top_n: int = 5,
    sim_threshold: float = 0.75,
) -> List[MergeSuggestion]:
    """Full merge suggestion pipeline.

    1. Compute centroids and pairwise similarity.
    2. Filter by similarity threshold.
    3. Evaluate silhouette delta for top candidates.
    4. Rank by silhouette improvement (positive delta preferred).

    Args:
        embeddings: Original 384-dim embeddings.
        embeddings_10d: 10-dim UMAP embeddings used for clustering.
        labels: HDBSCAN cluster labels.
        top_n: Maximum number of suggestions to return.
        sim_threshold: Minimum centroid similarity to consider.

    Returns:
        Ranked list of MergeSuggestion objects.
    """
    centroids = compute_cluster_centroids(embeddings, labels)
    candidates = compute_pairwise_cluster_similarity(centroids)

    # Filter by threshold
    candidates = [c for c in candidates if c.centroid_similarity >= sim_threshold]

    # Limit evaluation to a reasonable number
    candidates = candidates[: max(top_n * 3, 15)]

    suggestions: List[MergeSuggestion] = []
    for candidate in candidates:
        merge_score = evaluate_merge_silhouette(
            embeddings_10d, labels, candidate.cluster_a, candidate.cluster_b
        )
        silhouette_delta = merge_score.silhouette_delta if merge_score else 0.0

        size_a = int((labels == candidate.cluster_a).sum())
        size_b = int((labels == candidate.cluster_b).sum())
        coherence_a = _compute_intra_cluster_coherence(
            embeddings, labels, candidate.cluster_a
        )
        coherence_b = _compute_intra_cluster_coherence(
            embeddings, labels, candidate.cluster_b
        )

        suggestions.append(
            MergeSuggestion(
                cluster_a=candidate.cluster_a,
                cluster_b=candidate.cluster_b,
                cluster_a_size=size_a,
                cluster_b_size=size_b,
                centroid_similarity=round(candidate.centroid_similarity, 6),
                silhouette_delta=round(silhouette_delta, 6),
                coherence_a=round(coherence_a, 6),
                coherence_b=round(coherence_b, 6),
                rank=0,
            )
        )

    # Sort by silhouette delta (best improvement first), then by similarity
    suggestions.sort(
        key=lambda s: (s.silhouette_delta, s.centroid_similarity), reverse=True
    )

    # Assign ranks
    ranked: List[MergeSuggestion] = []
    for i, s in enumerate(suggestions[:top_n]):
        ranked.append(
            MergeSuggestion(
                cluster_a=s.cluster_a,
                cluster_b=s.cluster_b,
                cluster_a_size=s.cluster_a_size,
                cluster_b_size=s.cluster_b_size,
                centroid_similarity=s.centroid_similarity,
                silhouette_delta=s.silhouette_delta,
                coherence_a=s.coherence_a,
                coherence_b=s.coherence_b,
                rank=i + 1,
            )
        )

    return ranked
