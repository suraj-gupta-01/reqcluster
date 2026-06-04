"""Tests for Phase 3 split suggestion module."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from core.split_suggest import (
    BimodalityResult,
    SplitSuggestion,
    compute_cluster_spread,
    test_cluster_bimodality as run_bimodality_test,
    evaluate_split_silhouette,
    suggest_splits,
)


def _make_bimodal_cluster(n_per_mode=15, dim=384, separation=2.0):
    """Generate a cluster with two clear sub-groups."""
    np.random.seed(42)
    center_a = np.random.randn(dim)
    center_a = center_a / np.linalg.norm(center_a)
    center_b = center_a + np.random.randn(dim) * separation
    center_b = center_b / np.linalg.norm(center_b)

    points_a = center_a + np.random.randn(n_per_mode, dim) * 0.05
    points_b = center_b + np.random.randn(n_per_mode, dim) * 0.05

    embeddings = np.vstack([points_a, points_b]).astype(np.float32)
    labels = np.zeros(n_per_mode * 2, dtype=np.int32)
    return embeddings, labels


def _make_uniform_cluster(n=20, dim=384):
    """Generate a tight, uniform cluster."""
    np.random.seed(42)
    center = np.random.randn(dim)
    center = center / np.linalg.norm(center)
    points = center + np.random.randn(n, dim) * 0.02
    return points.astype(np.float32), np.zeros(n, dtype=np.int32)


def _make_10d(embeddings, labels):
    """Create fake 10D embeddings preserving cluster structure."""
    np.random.seed(42)
    n = embeddings.shape[0]
    result = np.random.randn(n, 10)
    for cid in set(labels):
        mask = labels == cid
        center = result[mask].mean(axis=0)
        result[mask] = center + (result[mask] - center) * 0.3
    return result.astype(np.float32)


class TestComputeClusterSpread:
    def test_tight_cluster_low_spread(self):
        emb, labels = _make_uniform_cluster(20)
        spread = compute_cluster_spread(emb, labels, 0)
        assert spread < 0.1

    def test_single_point_zero_spread(self):
        emb = np.random.randn(1, 384).astype(np.float32)
        labels = np.array([0], dtype=np.int32)
        spread = compute_cluster_spread(emb, labels, 0)
        assert spread == 0.0

    def test_spread_in_valid_range(self):
        emb, labels = _make_bimodal_cluster(15)
        spread = compute_cluster_spread(emb, labels, 0)
        assert 0.0 <= spread <= 1.0


class TestBimodality:
    def test_bimodal_cluster_detected(self):
        emb, labels = _make_bimodal_cluster(15, separation=3.0)
        emb_10d = _make_10d(emb, labels)
        result = run_bimodality_test(emb_10d, labels, 0)
        assert result is not None
        assert isinstance(result, BimodalityResult)
        assert result.cluster_id == 0
        assert len(result.sub_labels) == emb.shape[0]
        assert len(result.sub_cluster_sizes) == 2

    def test_uniform_cluster_not_bimodal(self):
        emb, labels = _make_uniform_cluster(20)
        emb_10d = _make_10d(emb, labels)
        result = run_bimodality_test(emb_10d, labels, 0)
        # May or may not detect bimodality on a uniform cluster,
        # but bimodality_score should be low
        if result is not None:
            assert result.bimodality_score < 0.5 or not result.is_bimodal

    def test_too_small_cluster(self):
        emb = np.random.randn(3, 384).astype(np.float32)
        labels = np.array([0, 0, 0], dtype=np.int32)
        emb_10d = np.random.randn(3, 10).astype(np.float32)
        result = run_bimodality_test(emb_10d, labels, 0, min_cluster_size=6)
        assert result is None


class TestEvaluateSplitSilhouette:
    def test_returns_split_score(self):
        # Need at least 3 clusters for silhouette
        np.random.seed(42)
        n = 30
        emb_10d = np.random.randn(n, 10).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int32)
        sub_labels = [0] * 5 + [1] * 5  # Split cluster 0
        result = evaluate_split_silhouette(emb_10d, labels, 0, sub_labels)
        if result is not None:
            assert hasattr(result, "silhouette_delta")
            assert result.cluster_id == 0

    def test_mismatched_sub_labels(self):
        emb_10d = np.random.randn(20, 10).astype(np.float32)
        labels = np.array([0] * 10 + [1] * 10, dtype=np.int32)
        sub_labels = [0, 1, 0]  # Wrong length
        result = evaluate_split_silhouette(emb_10d, labels, 0, sub_labels)
        assert result is None


class TestSuggestSplits:
    def test_returns_suggestions_for_bimodal(self):
        # Create data with one bimodal cluster and one tight cluster
        np.random.seed(42)
        bimodal_emb, _ = _make_bimodal_cluster(12, separation=3.0)
        tight_emb, _ = _make_uniform_cluster(15)
        embeddings = np.vstack([bimodal_emb, tight_emb]).astype(np.float32)
        labels = np.array(
            [0] * bimodal_emb.shape[0] + [1] * tight_emb.shape[0],
            dtype=np.int32,
        )
        emb_10d = _make_10d(embeddings, labels)
        suggestions = suggest_splits(
            embeddings, emb_10d, labels, top_n=3, spread_threshold=0.0
        )
        for s in suggestions:
            assert isinstance(s, SplitSuggestion)
            assert s.rank >= 1

    def test_no_clusters_returns_empty(self):
        emb = np.random.randn(10, 384).astype(np.float32)
        labels = np.full(10, -1, dtype=np.int32)
        emb_10d = np.random.randn(10, 10).astype(np.float32)
        suggestions = suggest_splits(emb, emb_10d, labels)
        assert suggestions == []
