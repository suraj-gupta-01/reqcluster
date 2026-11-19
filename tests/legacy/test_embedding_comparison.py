import json

import numpy as np

from core.embedding_comparison import compare_embeddings


def test_identical_arrays_are_json_serializable_and_similarity_near_one():
    embeddings = np.eye(3, dtype=np.float32)
    report = compare_embeddings(embeddings, embeddings)

    json.dumps(report)
    assert report["aggregate"]["mean_cosine_similarity"] == 1.0
    assert report["aggregate"]["mean_delta"] == 0.0


def test_changed_arrays_produce_nonzero_delta():
    base = np.eye(3, dtype=np.float32)
    candidate = np.roll(base, shift=1, axis=0)
    report = compare_embeddings(base, candidate)

    assert report["aggregate"]["mean_delta"] > 0.0
    assert report["delta_threshold_counts"]["0.05"] > 0


def test_topk_neighbor_preservation_handles_small_n():
    one = np.array([[1.0, 0.0]], dtype=np.float32)
    report = compare_embeddings(one, one, k=5)

    assert report["nearest_neighbor_preservation"]["k"] == 0
    assert report["nearest_neighbor_preservation"]["score"] is None
    assert report["warnings"]


def test_invalid_shapes_return_warnings():
    base = np.zeros((2, 3), dtype=np.float32)
    candidate = np.zeros((3, 3), dtype=np.float32)
    report = compare_embeddings(base, candidate)

    assert report["warnings"]
    assert report["per_requirement"] == []
