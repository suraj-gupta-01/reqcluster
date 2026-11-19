import numpy as np

from reqcluster.ml import labels as L
from reqcluster.ml import sampling


def blobs(n_per=30, centers=4, d=384, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(centers, d))
    X = np.vstack([c + 0.05 * rng.normal(size=(n_per, d)) for c in C])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), np.repeat(np.arange(centers), n_per)


def test_all_noise_and_single_cluster():
    X, _ = blobs()
    assert sampling.silhouette(X, np.full(len(X), -1)) is None
    assert sampling.silhouette(X, np.zeros(len(X), dtype=np.int64)) is None


def test_normalize_none_nan():
    out = L.normalize([None, float("nan"), 2, 0])
    assert out.dtype == np.int64 and out.tolist() == [-1, -1, 2, 0]


def test_silhouette_same_seed():
    X, y = blobs(n_per=40)
    a = sampling.silhouette(X, y, sample_size=60, seed=3)
    assert a == sampling.silhouette(X, y, sample_size=60, seed=3)


def test_knn_matches_brute_force():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 16)).astype(np.float32)
    idx, sim = sampling.knn(X, 5, block=64)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    expect = np.sort(S, axis=1)[:, ::-1][:, :5]
    assert np.allclose(sim, expect, atol=1e-5)
    assert (idx != np.arange(300)[:, None]).all()


def test_relabel_order_and_groups():
    out = L.relabel_contiguous(np.array([5, -1, 2, 5, 9]))
    assert out.tolist() == [0, -1, 1, 0, 2]
    assert L.cluster_ids(out) == [0, 1, 2]
    assert L.groups(out)[0].tolist() == [0, 3]
    ids, C = L.centroids(np.eye(5, dtype=np.float32), out)
    assert ids == [0, 1, 2] and C.shape == (3, 5)
