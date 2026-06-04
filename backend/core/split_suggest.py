"""Cluster split candidate detection via bimodality and silhouette analysis.

This module identifies clusters that may benefit from splitting based on:
- Intra-cluster spread (high spread → split candidate)
- Bimodality test using Gaussian Mixture Models (clear bimodal → split candidate)
- Silhouette score delta (positive delta → split improves coherence)

It does not modify any database state or run LLM calls.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


@dataclass(frozen=True)
class BimodalityResult:
    """Result of a bimodality test on a single cluster."""

    cluster_id: int
    is_bimodal: bool
    bic_1_component: float
    bic_2_component: float
    bic_improvement: float
    sub_cluster_sizes: List[int]
    sub_labels: List[int]
    bimodality_score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitScore:
    """Silhouette score delta for a hypothetical split."""

    cluster_id: int
    silhouette_before: float
    silhouette_after: float
    silhouette_delta: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SplitSuggestion:
    """A ranked split suggestion with scores and metadata."""

    cluster_id: int
    cluster_size: int
    spread_score: float
    bimodality_score: float
    bic_improvement: float
    silhouette_delta: float
    sub_cluster_sizes: List[int]
    sub_labels: List[int]
    rank: int

    def to_dict(self) -> dict:
        return asdict(self)


def compute_cluster_spread(
    embeddings: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
) -> float:
    """Compute intra-cluster distance variance (spread) for a cluster.

    Higher spread indicates a more dispersed cluster that may be a split candidate.

    Uses cosine distance (1 - cosine_similarity) from the cluster centroid.

    Returns:
        Float in [0, 1] range. 0 = perfectly tight, 1 = maximally dispersed.
    """
    mask = labels == cluster_id
    cluster_emb = embeddings[mask]
    n = cluster_emb.shape[0]
    if n < 2:
        return 0.0

    centroid = cluster_emb.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm

    # Cosine similarity to centroid
    norms = np.linalg.norm(cluster_emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = cluster_emb / norms
    similarities = normalized @ centroid

    # Cosine distance = 1 - similarity
    distances = 1.0 - similarities

    # Spread = standard deviation of distances, normalized
    spread = float(np.std(distances))
    # Clamp to [0, 1]
    return min(max(spread, 0.0), 1.0)


def test_cluster_bimodality(
    embeddings_10d: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    min_cluster_size: int = 6,
) -> Optional[BimodalityResult]:
    """Test whether a cluster has a bimodal distribution using GMM.

    Fits 1-component and 2-component Gaussian Mixture Models on the cluster's
    10D UMAP embeddings. A significant BIC improvement for 2 components
    indicates bimodality.

    Args:
        embeddings_10d: 10-dim UMAP embeddings.
        labels: HDBSCAN cluster labels.
        cluster_id: Cluster to test.
        min_cluster_size: Minimum points needed (must be >= 6 for 2-GMM).

    Returns:
        BimodalityResult or None if the cluster is too small.
    """
    mask = labels == cluster_id
    cluster_emb = embeddings_10d[mask]
    n = cluster_emb.shape[0]

    if n < min_cluster_size:
        return None

    try:
        gmm_1 = GaussianMixture(n_components=1, random_state=42, covariance_type="full")
        gmm_1.fit(cluster_emb)
        bic_1 = gmm_1.bic(cluster_emb)

        gmm_2 = GaussianMixture(n_components=2, random_state=42, covariance_type="full")
        gmm_2.fit(cluster_emb)
        bic_2 = gmm_2.bic(cluster_emb)
    except Exception:
        return None

    bic_improvement = float(bic_1 - bic_2)
    sub_labels_array = gmm_2.predict(cluster_emb)
    sub_labels = [int(l) for l in sub_labels_array]

    sub_sizes = [
        int((sub_labels_array == comp).sum()) for comp in range(2)
    ]

    # Bimodality score: normalized BIC improvement
    # Higher = more clearly bimodal
    # Use a sigmoid-like normalization so the score is in [0, 1]
    if bic_improvement <= 0:
        bimodality_score = 0.0
    else:
        # Scale: BIC improvement of ~100 → score ~0.5
        bimodality_score = min(1.0, bic_improvement / (bic_improvement + 100.0))

    # A cluster is considered bimodal if:
    # 1. BIC improvement is positive (2-component fits better)
    # 2. Both sub-clusters have at least 3 points
    is_bimodal = bic_improvement > 0 and min(sub_sizes) >= 3

    return BimodalityResult(
        cluster_id=cluster_id,
        is_bimodal=is_bimodal,
        bic_1_component=round(float(bic_1), 3),
        bic_2_component=round(float(bic_2), 3),
        bic_improvement=round(bic_improvement, 3),
        sub_cluster_sizes=sub_sizes,
        sub_labels=sub_labels,
        bimodality_score=round(bimodality_score, 6),
    )


def evaluate_split_silhouette(
    embeddings_10d: np.ndarray,
    labels: np.ndarray,
    cluster_id: int,
    sub_labels: List[int],
) -> Optional[SplitScore]:
    """Compute silhouette score delta if a cluster were split.

    A positive delta means the split improves overall clustering coherence.

    Args:
        embeddings_10d: 10-dim UMAP embeddings.
        labels: Current HDBSCAN cluster labels.
        cluster_id: Cluster being split.
        sub_labels: GMM sub-label assignments for points in the cluster.

    Returns:
        SplitScore or None if silhouette cannot be computed.
    """
    non_noise_mask = labels != -1
    filtered_embeddings = embeddings_10d[non_noise_mask]
    filtered_labels = labels[non_noise_mask].copy()

    if filtered_embeddings.shape[0] < 3:
        return None

    try:
        score_before = silhouette_score(filtered_embeddings, filtered_labels)
    except ValueError:
        return None

    # Create hypothetical split labels
    # The new sub-cluster gets max_label + 1
    split_labels = filtered_labels.copy()
    max_label = int(split_labels.max())
    new_label = max_label + 1

    # Map sub_labels onto the filtered label array
    cluster_indices_in_filtered = np.where(filtered_labels == cluster_id)[0]
    if len(cluster_indices_in_filtered) != len(sub_labels):
        return None

    for i, idx in enumerate(cluster_indices_in_filtered):
        if sub_labels[i] == 1:
            split_labels[idx] = new_label

    if len(set(split_labels)) < 2:
        return None

    try:
        score_after = silhouette_score(filtered_embeddings, split_labels)
    except ValueError:
        return None

    return SplitScore(
        cluster_id=cluster_id,
        silhouette_before=round(float(score_before), 6),
        silhouette_after=round(float(score_after), 6),
        silhouette_delta=round(float(score_after - score_before), 6),
    )


def suggest_splits(
    embeddings: np.ndarray,
    embeddings_10d: np.ndarray,
    labels: np.ndarray,
    top_n: int = 5,
    spread_threshold: float = 0.3,
) -> List[SplitSuggestion]:
    """Full split suggestion pipeline.

    1. Compute spread for each cluster.
    2. Test bimodality for clusters above spread threshold.
    3. Evaluate silhouette delta for bimodal clusters.
    4. Rank by bimodality score and silhouette improvement.

    Args:
        embeddings: Original 384-dim embeddings.
        embeddings_10d: 10-dim UMAP embeddings used for clustering.
        labels: HDBSCAN cluster labels.
        top_n: Maximum number of suggestions to return.
        spread_threshold: Minimum spread to consider for splitting.

    Returns:
        Ranked list of SplitSuggestion objects.
    """
    unique_clusters = sorted(set(int(l) for l in labels if l != -1))

    # Step 1: Compute spread for each cluster
    spread_scores: Dict[int, float] = {}
    for cid in unique_clusters:
        spread = compute_cluster_spread(embeddings, labels, cid)
        spread_scores[cid] = spread

    # Step 2: Filter by spread threshold and test bimodality
    suggestions: List[SplitSuggestion] = []
    for cid in unique_clusters:
        spread = spread_scores[cid]
        cluster_size = int((labels == cid).sum())

        # Even if below threshold, still test if cluster is large enough
        if spread < spread_threshold and cluster_size < 10:
            continue

        bimodality = test_cluster_bimodality(embeddings_10d, labels, cid)
        if bimodality is None:
            continue

        if not bimodality.is_bimodal:
            continue

        # Step 3: Evaluate silhouette delta
        split_score = evaluate_split_silhouette(
            embeddings_10d, labels, cid, bimodality.sub_labels
        )
        silhouette_delta = split_score.silhouette_delta if split_score else 0.0

        suggestions.append(
            SplitSuggestion(
                cluster_id=cid,
                cluster_size=cluster_size,
                spread_score=round(spread, 6),
                bimodality_score=bimodality.bimodality_score,
                bic_improvement=bimodality.bic_improvement,
                silhouette_delta=round(silhouette_delta, 6),
                sub_cluster_sizes=bimodality.sub_cluster_sizes,
                sub_labels=bimodality.sub_labels,
                rank=0,
            )
        )

    # Sort by bimodality score then silhouette delta
    suggestions.sort(
        key=lambda s: (s.bimodality_score, s.silhouette_delta), reverse=True
    )

    # Assign ranks
    ranked: List[SplitSuggestion] = []
    for i, s in enumerate(suggestions[:top_n]):
        ranked.append(
            SplitSuggestion(
                cluster_id=s.cluster_id,
                cluster_size=s.cluster_size,
                spread_score=s.spread_score,
                bimodality_score=s.bimodality_score,
                bic_improvement=s.bic_improvement,
                silhouette_delta=s.silhouette_delta,
                sub_cluster_sizes=s.sub_cluster_sizes,
                sub_labels=s.sub_labels,
                rank=i + 1,
            )
        )

    return ranked
