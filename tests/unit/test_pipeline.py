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
    assert (a.labels >= 0).all() and ((a.probabilities >= 0) & (a.probabilities <= 1)).all()
    assert len(set(a.probabilities.round(3).tolist())) > 1  # a real per-row score, not a constant
    assert 8 <= len(a.clusters) <= min(40, len(X) // 5)
    assert (a.labels == b.labels).all()


def test_ward_tiny_n_all_noise():
    X, t = rand(3)
    assert (pipeline.run(X, t, P(clusterer="ward"), seed=0).labels == -1).all()


def test_viz_epochs_drop_only_for_large_n():
    from reqcluster.ml.reduce import _viz_epochs

    assert _viz_epochs(10_000) == _viz_epochs(20_000) == 100
    assert _viz_epochs(35_000) == 50


def test_ward_mode_keeps_hdbscan_outliers(monkeypatch):
    # ward draws the groups, HDBSCAN still flags outliers (adversarial: ARI .413 vs .346 pure ward)
    from reqcluster.ml import cluster

    X, t = blobs()
    real = cluster.hdbscan

    def hdb_with_outliers(x10, mcs, ms):
        labels, probs = real(x10, mcs, ms)
        labels = labels.copy()
        labels[[3, 40, 77]] = -1  # rows HDBSCAN calls noise
        return labels, probs

    monkeypatch.setattr(cluster, "hdbscan", hdb_with_outliers)
    w = pipeline.run(X, t, P(clusterer="ward"), seed=3)
    assert set(np.flatnonzero(w.labels == -1)) == {3, 40, 77}  # same outliers as HDBSCAN
    assert (w.probabilities[w.labels == -1] == 0).all() and (w.probabilities[w.labels >= 0] > 0).any()
    assert len(w.clusters) >= 8  # groups still come from ward


def test_membership_is_centroid_silhouette():
    from reqcluster.ml.cluster import membership

    x = np.array([[0.0], [0.1], [0.9], [10.0], [10.1], [5.0]])
    lab = np.array([0, 0, 0, 1, 1, -1])
    m = membership(x, lab)
    assert m[5] == 0.0 and m[0] > m[2] > 0  # core row above edge row; noise none
    assert np.isclose(m[3], 1 - 0.05 / (10.0 - 1 / 3))
    assert (membership(x, np.array([0, 0, 0, 0, 0, -1]))[:5] == 1.0).all()
    assert (membership(x, np.full(6, -1)) == 0).all()


def test_consensus_keeps_agreed_groups_and_scores_agreement():
    from reqcluster.ml.cluster import consensus

    X = np.vstack([np.eye(3)[[0]].repeat(6, 0), np.eye(3)[[1]].repeat(6, 0), np.eye(3)[[2]]])
    a = np.array([0] * 6 + [1] * 6 + [-1])
    b = a.copy()
    b[5] = 1  # one run disagrees about row 5
    labels, probs = consensus([a, a, a, a, b], 3, 1, X)
    assert len(set(labels[:6].tolist())) == 1 and len(set(labels[6:12].tolist())) == 1
    assert labels[0] != labels[6] and labels[12] == -1 and probs[12] == 0
    assert probs[0] > probs[5] > 0.5  # row 5 is where one run disagreed


def test_rescue_only_within_cluster_radius():
    from reqcluster.ml.cluster import rescue

    X = np.array([[1, 0.0], [1, 0.2], [1, -0.2], [1, 0.1], [0, 1.0]])
    lab = np.array([0, 0, 0, -1, -1])
    assert rescue(X, lab).tolist() == [0, 0, 0, 0, -1]  # inside the radius joins; far row stays


def test_small_runs_use_seed_consensus(monkeypatch):
    from reqcluster.ml import cluster

    seen = []
    real = cluster.consensus
    monkeypatch.setattr(cluster, "consensus", lambda runs, *a: seen.append(len(runs)) or real(runs, *a))
    X, t = blobs()
    a = pipeline.run(X, t, P(), seed=7)
    assert seen == [pipeline.CONSENSUS_RUNS]
    assert (a.labels == pipeline.run(X, t, P(), seed=7).labels).all()  # deterministic per seed
    pipeline.run(X, t, P(clusterer="ward"), seed=7)
    pipeline.run(*rand(pipeline.CONSENSUS_MIN - 1), P(), seed=7)
    assert len(seen) == 2  # ward and tiny sets skip consensus


def test_exact_duplicates_cluster_as_unique_rows():
    base, _ = two_blobs(n_per=20, seed=3)
    dup = np.repeat(base, 5, axis=0)
    full = pipeline.run(dup, [f"t{i}" for i in range(len(dup))], P(), seed=0)
    uniq = pipeline.run(base, [f"t{i}" for i in range(len(base))], P(), seed=0)
    for f in ("labels", "probabilities", "x10", "xy"):
        a = np.asarray(getattr(full, f)).reshape(len(base), 5, -1)
        assert (a == a[:, :1]).all()
    assert (np.asarray(full.labels)[::5] == np.asarray(uniq.labels)).all()
