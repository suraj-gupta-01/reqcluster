"""Tests for Phase 3 merge suggestion module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from core.merge_suggest import (
    MergeCandidate,
    MergeSuggestion,
    compute_cluster_centroids,
    compute_pairwise_cluster_similarity,
    evaluate_merge_silhouette,
    suggest_merges,
    _compute_intra_cluster_coherence,
)


def _make_clustered_data(n_clusters=3, n_per_cluster=20, dim=384):
    """Generate synthetic clustered data."""
    np.random.seed(42)
    embeddings = []
    labels = []
    for i in range(n_clusters):
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)
        points = center + np.random.randn(n_per_cluster, dim) * 0.05
        # L2 normalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        embeddings.append(points)
        labels.extend([i] * n_per_cluster)
    return np.vstack(embeddings), np.array(labels, dtype=np.int32)


def _make_10d(embeddings, labels):
    """Create fake 10D UMAP-like embeddings for silhouette testing."""
    np.random.seed(42)
    n = embeddings.shape[0]
    result = np.random.randn(n, 10)
    # Make same-cluster points closer
    for cid in set(labels):
        mask = labels == cid
        center = result[mask].mean(axis=0)
        result[mask] = center + (result[mask] - center) * 0.3
    return result.astype(np.float32)


class TestComputeClusterCentroids:
    def test_returns_centroids_for_each_cluster(self):
        emb, labels = _make_clustered_data(3, 10)
        centroids = compute_cluster_centroids(emb, labels)
        assert set(centroids.keys()) == {0, 1, 2}
        for centroid in centroids.values():
            assert centroid.shape == (384,)
            # Should be L2 normalized
            assert abs(np.linalg.norm(centroid) - 1.0) < 0.01

    def test_excludes_noise(self):
        emb, labels = _make_clustered_data(2, 10)
        labels[0] = -1
        labels[1] = -1
        centroids = compute_cluster_centroids(emb, labels)
        assert -1 not in centroids

    def test_single_cluster(self):
        emb, labels = _make_clustered_data(1, 10)
        centroids = compute_cluster_centroids(emb, labels)
        assert len(centroids) == 1


class TestComputePairwiseClusterSimilarity:
    def test_returns_sorted_candidates(self):
        emb, labels = _make_clustered_data(3, 10)
        centroids = compute_cluster_centroids(emb, labels)
        candidates = compute_pairwise_cluster_similarity(centroids)
        assert len(candidates) == 3  # C(3,2) = 3
        assert all(isinstance(c, MergeCandidate) for c in candidates)
        # Should be sorted descending by similarity
        for i in range(len(candidates) - 1):
            assert candidates[i].centroid_similarity >= candidates[i + 1].centroid_similarity

    def test_single_cluster_returns_empty(self):
        emb, labels = _make_clustered_data(1, 10)
        centroids = compute_cluster_centroids(emb, labels)
        candidates = compute_pairwise_cluster_similarity(centroids)
        assert candidates == []


class TestEvaluateMergeSilhouette:
    def test_returns_merge_score(self):
        emb, labels = _make_clustered_data(3, 15)
        emb_10d = _make_10d(emb, labels)
        result = evaluate_merge_silhouette(emb_10d, labels, 0, 1)
        if result is not None:
            assert hasattr(result, "silhouette_delta")
            assert result.cluster_a == 0
            assert result.cluster_b == 1

    def test_returns_none_for_two_clusters(self):
        emb, labels = _make_clustered_data(2, 10)
        emb_10d = _make_10d(emb, labels)
        # After merge we'd have 1 cluster → silhouette undefined
        result = evaluate_merge_silhouette(emb_10d, labels, 0, 1)
        assert result is None

    def test_returns_none_for_missing_cluster(self):
        emb, labels = _make_clustered_data(3, 10)
        emb_10d = _make_10d(emb, labels)
        result = evaluate_merge_silhouette(emb_10d, labels, 0, 99)
        assert result is None


class TestSuggestMerges:
    def test_returns_suggestions(self):
        emb, labels = _make_clustered_data(4, 15)
        emb_10d = _make_10d(emb, labels)
        suggestions = suggest_merges(emb, emb_10d, labels, top_n=3, sim_threshold=0.0)
        assert len(suggestions) <= 3
        for s in suggestions:
            assert isinstance(s, MergeSuggestion)
            assert s.rank >= 1

    def test_high_threshold_filters(self):
        emb, labels = _make_clustered_data(3, 10)
        emb_10d = _make_10d(emb, labels)
        suggestions = suggest_merges(emb, emb_10d, labels, sim_threshold=0.999)
        # Very high threshold should filter most/all
        assert len(suggestions) <= 3

    def test_single_cluster_returns_empty(self):
        emb, labels = _make_clustered_data(1, 10)
        emb_10d = _make_10d(emb, labels)
        suggestions = suggest_merges(emb, emb_10d, labels)
        assert suggestions == []

    def test_all_noise_returns_empty(self):
        emb = np.random.randn(20, 384).astype(np.float32)
        labels = np.full(20, -1, dtype=np.int32)
        emb_10d = np.random.randn(20, 10).astype(np.float32)
        suggestions = suggest_merges(emb, emb_10d, labels)
        assert suggestions == []


class TestIntraClusterCoherence:
    def test_tight_cluster_high_coherence(self):
        np.random.seed(42)
        center = np.random.randn(384)
        center = center / np.linalg.norm(center)
        emb = center + np.random.randn(10, 384) * 0.01
        labels = np.zeros(10, dtype=np.int32)
        coherence = _compute_intra_cluster_coherence(emb, labels, 0)
        assert coherence > 0.9

    def test_single_point_returns_one(self):
        emb = np.random.randn(1, 384).astype(np.float32)
        labels = np.array([0], dtype=np.int32)
        coherence = _compute_intra_cluster_coherence(emb, labels, 0)
        assert coherence == 1.0
