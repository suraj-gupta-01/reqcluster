"""Tests for Phase 5 active-learning core: constraints, uncertainty, quality."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from core.constrained_clustering import apply_constraints
from core.active_learning import uncertainty_queue, clustering_quality


def _emb(n, dim=8):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_must_link_merges_points_into_one_cluster():
    labels = np.array([0, 0, 1, 1, 2])
    emb = _emb(5)
    new, info = apply_constraints(emb, labels, None, must_links=[(0, 2)], cannot_links=[])
    # point 0 and point 2 must end up in the same cluster.
    assert new[0] == new[2]
    assert info["points_moved_must_link"] >= 1


def test_cannot_link_separates_co_clustered_points():
    labels = np.array([0, 0, 0, 1, 1])
    probs = np.array([0.9, 0.4, 0.95, 0.9, 0.9])
    emb = _emb(5)
    new, info = apply_constraints(emb, labels, probs, must_links=[], cannot_links=[(0, 1)])
    # points 0 and 1 must not share a cluster anymore.
    assert new[0] != new[1]
    assert info["points_moved_cannot_link"] == 1


def test_cannot_link_moves_lower_confidence_point():
    labels = np.array([0, 0, 1])
    probs = np.array([0.95, 0.30, 0.9])  # point 1 is least confident
    emb = _emb(3)
    new, _ = apply_constraints(emb, labels, probs, must_links=[], cannot_links=[(0, 1)])
    assert new[0] == 0          # high-confidence keeper stays
    assert new[1] != 0          # low-confidence loser moves


def test_uncertainty_queue_orders_noise_and_low_prob_first():
    labels = np.array([0, 0, -1, 1])
    probs = np.array([0.95, 0.50, 0.0, 0.99])
    texts = ["a", "b", "c", "d"]
    req_ids = ["R1", "R2", "R3", "R4"]
    q = uncertainty_queue(labels, probs, texts, req_ids, top_k=3)
    assert len(q) == 3
    # Noise (R3) is most uncertain, then the 0.50 membership (R2).
    assert q[0]["req_id"] == "R3"
    assert q[1]["req_id"] == "R2"


def test_clustering_quality_reports_metrics():
    labels = np.array([0, 0, 0, 1, 1, 1, -1])
    emb = _emb(7)
    q = clustering_quality(emb, labels)
    assert q["n_clusters"] == 2
    assert q["noise_count"] == 1
    assert 0.0 <= q["noise_rate"] <= 1.0


def test_no_constraints_is_noop():
    labels = np.array([0, 1, 2])
    emb = _emb(3)
    new, info = apply_constraints(emb, labels, None, must_links=[], cannot_links=[])
    assert np.array_equal(new, labels)
    assert info["points_moved_must_link"] == 0
    assert info["points_moved_cannot_link"] == 0
