import json
import re
from dataclasses import asdict

import numpy as np

from reqcluster.domain import ClusteringParams
from reqcluster.ml.analysis import compare_embeddings, run_ablation


def test_identical_arrays_are_json_serializable_and_similarity_near_one():
    e = np.eye(3, dtype=np.float32)
    r = compare_embeddings(e, e)
    json.dumps(asdict(r))
    assert r.aggregate.mean_cosine_similarity == 1.0
    assert r.aggregate.mean_delta == 0.0


def test_changed_arrays_produce_nonzero_delta():
    base = np.eye(3, dtype=np.float32)
    r = compare_embeddings(base, np.roll(base, 1, axis=0))
    assert r.aggregate.mean_delta > 0.0
    assert r.delta_threshold_counts["0.05"] > 0


def test_topk_neighbor_preservation_handles_small_n():
    one = np.array([[1.0, 0.0]], dtype=np.float32)
    r = compare_embeddings(one, one, k=5)
    assert r.nearest_neighbor_preservation.k == 0
    assert r.nearest_neighbor_preservation.score is None
    assert r.warnings


def test_invalid_shapes_return_warnings():
    r = compare_embeddings(np.zeros((2, 3)), np.zeros((3, 3)))
    assert r.warnings
    assert r.per_requirement == []


def _blob_embed(texts):
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(3, 32))
    out = []
    for t in texts:
        i = int(re.search(r"t(\d+)", t).group(1))
        out.append(centers[i % 3] + np.random.default_rng(i).normal(size=32) * 0.05)
    return np.asarray(out, dtype=np.float32)


def test_ablation_returns_base_enriched_sections():
    texts = [f"t{i}" for i in range(36)]
    params = ClusteringParams(embedding_mode="hybrid", min_cluster_size=5)
    r = run_ablation(texts, ["ctx"] * 36, params, _blob_embed, seed=1)
    json.dumps(asdict(r))
    assert r.mode == "hybrid"
    assert r.base.embedding_shape == [36, 32]
    assert r.enriched.n_clusters >= 1
    assert r.comparison.n_requirements == 36


def test_ablation_handles_small_datasets_with_warnings():
    params = ClusteringParams(embedding_mode="hybrid")
    r = run_ablation(["t0", "t1"], [None, None], params, _blob_embed, seed=1)
    assert r.base.n_clusters == 0
    assert r.enriched.noise_count == 2
    assert r.warnings
