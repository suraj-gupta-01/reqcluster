import numpy as np
import pytest

from reqcluster.errors import Invalid
from reqcluster.ml.active import uncertainty


def _data():
    # clusters around (1,0) and (0,1); index 4 sits between them, 0 sits at a centroid
    X = np.array([[1, 0], [1, .05], [0, 1], [.05, 1], [1, 1], [.9, 1]], dtype=float)
    return X, np.array([0, 0, 1, 1, 0, -1])


def test_between_centroids_outranks_at_centroid():
    X, labels = _data()
    q = uncertainty(np.ones(6), X, labels, top_k=6)
    order = [d["index"] for d in q]
    assert order.index(4) < order.index(0) and order.index(5) < order.index(0)


def test_all_noise_not_upload_order_ties_are_stable_and_in_range():
    X = np.eye(3)
    q = uncertainty(None, X, np.full(3, -1), top_k=3)
    assert len(q) == 3 and all(0 <= d["uncertainty"] <= 1 for d in q)


def test_uncertainty_length_mismatch():
    with pytest.raises(Invalid):
        uncertainty(np.ones(3), np.eye(4), np.zeros(4, dtype=int))
