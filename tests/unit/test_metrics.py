import numpy as np

from reqcluster.ml.metrics import compute

COORDS = np.array([[0, 0], [0, 1], [1, 0], [9, 9], [9, 8], [8, 9]], dtype=np.float32)


def test_perfect_clustering_scores_max():
    m = compute(np.array([0, 0, 0, 1, 1, 1]), COORDS, 0, ["A", "A", "A", "B", "B", "B"])
    assert m["has_ground_truth"] is True
    assert m["purity"] == 1.0
    assert m["ari"] == 1.0
    assert m["v_measure"] == 1.0
    assert m["hungarian_accuracy"] == 1.0
    assert m["accuracy_noise_inclusive"] == 1.0
    assert m["n_clusters"] == 2
    assert m["noise_count"] == 0


def test_noise_and_coverage():
    m = compute(np.array([0, 0, 1, 1, -1]), None, 0)
    assert m["n"] == 5
    assert m["noise_count"] == 1
    assert m["noise_rate"] == 0.2
    assert m["coverage"] == 0.8


def test_no_ground_truth_returns_intrinsic_only():
    m = compute(np.array([0, 0, 1, 1]), COORDS[[0, 1, 3, 4]], 0)
    assert m["has_ground_truth"] is False
    assert "purity" not in m
    assert m["silhouette"] is not None


def test_single_generic_label_is_not_ground_truth():
    m = compute(np.array([0, 0, 1, 1]), COORDS[[0, 1, 3, 4]], 0, ["General", "", "General", ""])
    assert m["has_ground_truth"] is False


def test_noise_counts_as_wrong_in_inclusive_accuracy():
    labels = np.array([0, 0, 0, 1, 1, -1, -1, -1])
    gt = ["A", "A", "A", "B", "B", "B", "A", "B"]
    x = np.zeros((8, 2), dtype=np.float32)
    m = compute(labels, x, 0, gt)
    assert m["hungarian_accuracy"] == 1.0  # non-noise points all matched
    assert m["accuracy_noise_inclusive"] == 5 / 8


def test_single_cluster_hungarian_matches_largest_reference_group():
    m = compute(np.array([0, 0, 0, 0]), None, 0, ["A", "A", "B", "B"])
    assert m["hungarian_accuracy"] == 0.5
    assert m["accuracy_noise_inclusive"] == 0.5
