import numpy as np
import pytest

from reqcluster.domain import ClusteringParams
from reqcluster.errors import Invalid
from reqcluster.ml import pipeline


def two_blobs(n_per=30, d=384, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(2, d))
    X = np.vstack([c + 0.05 * rng.normal(size=(n_per, d)) for c in C])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    texts = [f"alpha brake{i}" for i in range(n_per)] + [f"beta display{i}" for i in range(n_per)]
    return X.astype(np.float32), texts


def P(**kw):
    return ClusteringParams(**kw)


def rand(n, d=384, seed=0):
    X = np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)
    return X / np.linalg.norm(X, axis=1, keepdims=True), [f"req {i}" for i in range(n)]


@pytest.mark.parametrize("n", [1, 3, 8])
def test_small_n_completes(n):
    X, t = rand(n)
    r = pipeline.run(X, t, P(), seed=0)
    assert r.xy.shape == (n, 2) and len(r.labels) == n
    if n < 5:
        assert (r.labels == -1).all()
        assert "fewer than 5 requirements: clustering skipped" in r.warnings


def test_big_min_cluster_size_all_noise():
    X, t = rand(12)
    r = pipeline.run(X, t, P(min_cluster_size=60), seed=0)
    assert (r.labels == -1).all()
    assert "all requirements were classified as noise" in r.warnings


def test_two_blobs_and_determinism():
    X, t = two_blobs()
    steps = []
    a = pipeline.run(X, t, P(min_cluster_size=5), seed=7, progress=lambda f, s: steps.append(s))
    b = pipeline.run(X, t, P(min_cluster_size=5), seed=7)
    assert len(a.clusters) == 2
    assert np.array_equal(a.labels, b.labels) and np.array_equal(a.x10, b.x10)
    assert steps[:4] == ["reduce", "cluster", "keywords", "graph"]
    assert set(a.durations) == {"reduce", "cluster", "keywords", "graph"}
    assert all(c.cluster_id != -1 for c in a.clusters)
    for i, j, _s in a.edges:
        assert i < j and a.labels[i] != -1 and a.labels[j] != -1


def test_length_mismatch():
    X, t = rand(10)
    with pytest.raises(Invalid):
        pipeline.run(X, t[:5], P(), seed=0)
    with pytest.raises(Invalid):
        pipeline.run(X[:0], [], P(), seed=0)


def test_reduce_never_runs_umap_concurrently(monkeypatch):
    # numba's default workqueue threading layer aborts the process on concurrent use (seen in Docker)
    import threading
    import time

    import umap

    from reqcluster.ml import reduce

    active, peak, lock = [0], [0], threading.Lock()

    class FakeUMAP:
        def __init__(self, n_components=2, **kw):
            self.k = n_components

        def fit_transform(self, X):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1
            return np.zeros((len(X), self.k), np.float32)

    monkeypatch.setattr(umap, "UMAP", FakeUMAP)
    X = rand(40)[0]
    ts = [threading.Thread(target=reduce.reduce, args=(X, 0)) for _ in range(3)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert peak[0] == 1


def test_graph_keeps_edges_for_loose_group_under_small_budget():
    from reqcluster.ml import graph

    rng = np.random.default_rng(1)
    d = 32
    groups = []
    for noise, n in ((0.01, 12), (0.01, 12), (0.2, 6)):
        c = rng.normal(size=d)
        groups.append(c + noise * rng.normal(size=(n, d)))
    X = np.vstack(groups)
    labels = np.repeat([0, 1, 2], [12, 12, 6])
    edges = graph.build(X, labels, threshold=0.3, k=5, max_edges=50)
    assert len(edges) <= 50
    assert any(a >= 24 and b >= 24 for a, b, _ in edges)


def test_exact_knn_matches_brute_force_with_self_first():
    from reqcluster.ml.reduce import _exact_knn

    X = rand(120, d=32, seed=3)[0]
    idx, dist, _ = _exact_knn(X, 15)
    D = 1.0 - X @ X.T
    assert idx.shape == dist.shape == (120, 15)
    assert (idx[:, 0] == np.arange(120)).all() and np.allclose(dist[:, 0], 0)
    for i in range(120):
        assert np.allclose(np.sort(D[i])[:15], dist[i], atol=1e-5)  # ascending, exact


def blobs(k=12, per=15, d=384, seed=0):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(k, d))
    X = np.vstack([c + 0.05 * rng.normal(size=(per, d)) for c in C])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32), [f"req {i}" for i in range(len(X))]


def test_ward_no_noise_k_in_range_and_deterministic():
    X, t = blobs()
    a = pipeline.run(X, t, P(clusterer="ward"), seed=3)
    b = pipeline.run(X, t, P(clusterer="ward"), seed=3)
    assert (a.labels >= 0).all() and (a.probabilities == 1.0).all()
    assert 8 <= len(a.clusters) <= min(40, len(X) // 5)
    assert (a.labels == b.labels).all()


def test_ward_tiny_n_all_noise():
    X, t = rand(3)
    assert (pipeline.run(X, t, P(clusterer="ward"), seed=0).labels == -1).all()
