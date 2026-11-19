import numpy as np
import pytest

from reqcluster.errors import Invalid
from reqcluster.ml.active import quality, uncertainty


def _emb(n, dim=8):
    x = np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def test_uncertainty_orders_noise_and_low_prob_first():
    labels = np.array([0, 0, -1, 1])
    q = uncertainty(np.array([0.95, 0.5, 0.0, 0.99]), _emb(4), labels, top_k=3)
    assert len(q) == 3
    assert [d["index"] for d in q[:2]] == [2, 1]


def test_uncertainty_length_mismatch():
    with pytest.raises(Invalid):
        uncertainty(np.ones(3), _emb(4), np.zeros(4, dtype=int))


def test_quality_reports_metrics():
    q = quality(_emb(7), np.array([0, 0, 0, 1, 1, 1, -1]), seed=0)
    assert q["n_clusters"] == 2 and q["noise_count"] == 1
    assert 0.0 <= q["noise_rate"] <= 1.0


def test_quality_single_cluster_has_no_silhouette():
    assert quality(_emb(3), np.zeros(3, dtype=int), seed=0)["silhouette"] is None
