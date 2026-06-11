"""Tests for clustering validation metrics."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from core.metrics import compute_metrics


def test_perfect_clustering_scores_max():
    labels = [0, 0, 0, 1, 1, 1]
    gt = ["A", "A", "A", "B", "B", "B"]
    coords = [[0, 0], [0, 1], [1, 0], [9, 9], [9, 8], [8, 9]]
    m = compute_metrics(labels, gt, coords)
    assert m["has_ground_truth"] is True
    assert m["accuracy_pct"] == 100.0
    assert m["ari"] == 1.0
    assert m["v_measure"] == 1.0
    assert m["n_clusters"] == 2
    assert m["noise"] == 0


def test_noise_and_coverage():
    labels = [0, 0, 1, 1, -1]
    m = compute_metrics(labels, None, None)
    assert m["n"] == 5
    assert m["noise"] == 1
    assert m["noise_pct"] == 20.0
    assert m["coverage_pct"] == 80.0


def test_no_ground_truth_returns_intrinsic_only():
    labels = [0, 0, 1, 1]
    coords = [[0, 0], [0, 1], [9, 9], [9, 8]]
    m = compute_metrics(labels, None, coords)
    assert m["has_ground_truth"] is False
    assert "accuracy_pct" not in m
    assert "silhouette" in m  # intrinsic still computed


def test_single_generic_label_is_not_ground_truth():
    # 'General' / blank labels do not count as ground truth.
    labels = [0, 0, 1, 1]
    gt = ["General", "", "General", ""]
    m = compute_metrics(labels, gt, [[0, 0], [0, 1], [9, 9], [9, 8]])
    assert m["has_ground_truth"] is False
