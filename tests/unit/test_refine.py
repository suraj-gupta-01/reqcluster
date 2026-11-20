import numpy as np

from reqcluster.ml import refine as R


def _clustered(n_clusters=3, n_per=20, dim=384):
    rng = np.random.default_rng(42)
    pts, labels = [], []
    for i in range(n_clusters):
        c = rng.standard_normal(dim)
        c /= np.linalg.norm(c)
        p = c + rng.standard_normal((n_per, dim)) * 0.05
        pts.append(p / np.linalg.norm(p, axis=1, keepdims=True))
        labels += [i] * n_per
    return np.vstack(pts), np.array(labels, dtype=np.int32)


def _10d(labels):
    rng = np.random.default_rng(42)
    r = rng.standard_normal((len(labels), 10))
    for cid in set(labels.tolist()):
        m = labels == cid
        c = r[m].mean(axis=0)
        r[m] = c + (r[m] - c) * 0.3
    return r.astype(np.float32)


def _bimodal(n_per=15, dim=384, sep=2.0):
    rng = np.random.default_rng(42)
    a = rng.standard_normal(dim)
    a /= np.linalg.norm(a)
    b = a + rng.standard_normal(dim) * sep
    b /= np.linalg.norm(b)
    e = np.vstack([a + rng.standard_normal((n_per, dim)) * 0.05,
                   b + rng.standard_normal((n_per, dim)) * 0.05]).astype(np.float32)
    return e, np.zeros(n_per * 2, dtype=np.int32)


def _uniform(n=20, dim=384):
    rng = np.random.default_rng(42)
    c = rng.standard_normal(dim)
    c /= np.linalg.norm(c)
    return (c + rng.standard_normal((n, dim)) * 0.02).astype(np.float32), np.zeros(n, dtype=np.int32)


# --- merge ---

def test_centroids_each_cluster_unit_norm():
    emb, labels = _clustered(3, 10)
    c = R.unit_centroids(emb, labels)
    assert set(c) == {0, 1, 2}
    for v in c.values():
        assert v.shape == (384,)
        assert abs(np.linalg.norm(v) - 1.0) < 0.01


def test_centroids_exclude_noise_and_single():
    emb, labels = _clustered(2, 10)
    labels[:2] = -1
    assert -1 not in R.unit_centroids(emb, labels)
    emb, labels = _clustered(1, 10)
    assert len(R.unit_centroids(emb, labels)) == 1
    assert R.pairwise_similarity(R.unit_centroids(emb, labels)) == []


def test_pairwise_sorted():
    emb, labels = _clustered(3, 10)
    c = R.pairwise_similarity(R.unit_centroids(emb, labels))
    assert len(c) == 3
    assert all(isinstance(x, R.MergeCandidate) for x in c)
    assert all(c[i].centroid_similarity >= c[i + 1].centroid_similarity for i in range(2))


def test_evaluate_merge():
    emb, labels = _clustered(3, 15)
    r = R.evaluate_merge(_10d(labels), labels, 0, 1)
    if r is not None:
        assert r.cluster_a == 0 and r.cluster_b == 1
    emb, labels = _clustered(2, 10)
    assert R.evaluate_merge(_10d(labels), labels, 0, 1) is None
    emb, labels = _clustered(3, 10)
    assert R.evaluate_merge(_10d(labels), labels, 0, 99) is None


def test_suggest_merges():
    emb, labels = _clustered(4, 15)
    s = R.suggest_merges(emb, _10d(labels), labels, top_n=3, sim_threshold=0.0)
    assert len(s) <= 3
    assert all(isinstance(x, R.MergeSuggestion) and x.rank >= 1 for x in s)
    emb, labels = _clustered(3, 10)
    assert len(R.suggest_merges(emb, _10d(labels), labels, sim_threshold=0.999)) <= 3
    emb, labels = _clustered(1, 10)
    assert R.suggest_merges(emb, _10d(labels), labels) == []
    rng = np.random.default_rng(0)
    assert R.suggest_merges(rng.standard_normal((20, 384)).astype(np.float32),
                            rng.standard_normal((20, 10)).astype(np.float32),
                            np.full(20, -1, dtype=np.int32)) == []


def test_coherence():
    rng = np.random.default_rng(42)
    c = rng.standard_normal(384)
    c /= np.linalg.norm(c)
    emb = c + rng.standard_normal((10, 384)) * 0.01
    assert R.coherence(emb, np.zeros(10, dtype=np.int32), 0) > 0.9
    assert R.coherence(emb[:1], np.array([0], dtype=np.int32), 0) == 1.0


# --- split ---

def test_spread():
    emb, labels = _uniform(20)
    assert R.cluster_spread(emb, labels, 0) < 0.1
    assert R.cluster_spread(emb[:1], np.array([0], dtype=np.int32), 0) == 0.0
    emb, labels = _bimodal(15)
    assert 0.0 <= R.cluster_spread(emb, labels, 0) <= 1.0


def test_bimodal_detected():
    emb, labels = _bimodal(15, sep=3.0)
    r = R.bimodality(_10d(labels), labels, 0)
    assert isinstance(r, R.BimodalityResult)
    assert r.cluster_id == 0
    assert len(r.sub_labels) == len(emb)
    assert len(r.sub_cluster_sizes) == 2


def test_uniform_not_bimodal_and_too_small():
    emb, labels = _uniform(20)
    r = R.bimodality(_10d(labels), labels, 0)
    if r is not None:
        assert r.bimodality_score < 0.5 or not r.is_bimodal
    rng = np.random.default_rng(0)
    assert R.bimodality(rng.standard_normal((3, 10)).astype(np.float32),
                        np.zeros(3, dtype=np.int32), 0, min_cluster_size=6) is None


def test_evaluate_split():
    rng = np.random.default_rng(42)
    x = rng.standard_normal((30, 10)).astype(np.float32)
    labels = np.array([0] * 10 + [1] * 10 + [2] * 10, dtype=np.int32)
    r = R.evaluate_split(x, labels, 0, [0] * 5 + [1] * 5)
    if r is not None:
        assert r.cluster_id == 0
    labels = np.array([0] * 10 + [1] * 10, dtype=np.int32)
    assert R.evaluate_split(x[:20], labels, 0, [0, 1, 0]) is None


def _split_data():
    b, _ = _bimodal(12, sep=3.0)
    t, _ = _uniform(15)
    emb = np.vstack([b, t]).astype(np.float32)
    labels = np.array([0] * len(b) + [1] * len(t), dtype=np.int32)
    return emb, _10d(labels), labels


def test_suggest_splits_and_deterministic():
    emb, x10, labels = _split_data()
    a = R.suggest_splits(emb, x10, labels, top_n=3, spread_threshold=0.0, seed=7)
    b = R.suggest_splits(emb, x10, labels, top_n=3, spread_threshold=0.0, seed=7)
    assert all(isinstance(s, R.SplitSuggestion) and s.rank >= 1 for s in a)
    assert a == b


def test_suggest_splits_no_clusters():
    rng = np.random.default_rng(0)
    assert R.suggest_splits(rng.standard_normal((10, 384)).astype(np.float32),
                            rng.standard_normal((10, 10)).astype(np.float32),
                            np.full(10, -1, dtype=np.int32)) == []


# --- representatives ---

def test_representatives_ranked_by_centroid_similarity():
    emb, labels = _clustered(2, 10, dim=16)
    labels[0] = -1
    texts = [f"r{i}" for i in range(20)]
    ids = [f"REQ-{i}" for i in range(20)]
    r = R.representatives(emb, texts, labels, ids, top_n=3)
    assert set(r) == {0, 1}
    assert [x.rank for x in r[1]] == [1, 2, 3]
    sims = [x.similarity_to_centroid for x in r[1]]
    assert sims == sorted(sims, reverse=True)
    assert all(labels[x.index] == 1 and x.text == texts[x.index] for x in r[1])


def test_suggest_splits_drops_non_improving(monkeypatch):
    emb, labels = _uniform(20)
    sub = [int(v) for v in np.random.default_rng(1).integers(0, 2, 20)]
    monkeypatch.setattr(R, "bimodality", lambda *a, **k: R.BimodalityResult(
        0, True, 0.0, 0.0, 50.0, [sub.count(0), sub.count(1)], sub, 0.3))
    assert R.suggest_splits(emb, _10d(labels), labels) == []


def test_coherence_assessment_calibrated_for_minilm():
    rng = np.random.default_rng(0)
    c = rng.standard_normal(384)
    c /= np.linalg.norm(c)
    emb = (c + rng.standard_normal((40, 384)) * np.sqrt(1.083 / 384)).astype(np.float32)
    r = R.score_coherence(emb, np.zeros(40, dtype=np.int32), 0, [])
    assert 0.42 < r["coherence_score"] < 0.54
    assert not r["assessment"].startswith(("Low", "Very low"))


def test_coherence_bands_follow_the_embedding_model(monkeypatch):
    from reqcluster import embeddings
    rng = np.random.default_rng(0)
    base = rng.standard_normal(384)
    E = base + 0.9 * rng.standard_normal((40, 384))  # mean pairwise cosine ~ .55
    E = (E / np.linalg.norm(E, axis=1, keepdims=True)).astype(np.float32)
    lab = np.zeros(40, dtype=np.int32)
    assert R.score_coherence(E, lab, 0, [])["assessment"].startswith("Highly")  # MiniLM bands
    monkeypatch.setattr(embeddings, "calibration",
                        lambda model=None: embeddings.CALIBRATION["Alibaba-NLP/gte-modernbert-base"])
    assert R.score_coherence(E, lab, 0, [])["assessment"].startswith("Very low")
