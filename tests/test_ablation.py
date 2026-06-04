import json

import numpy as np

from core import ablation
from core.domain_embeddings import EmbeddingMode


def fake_embeddings(texts, enriched_texts=None, config=None, **kwargs):
    n = len(texts)
    arr = np.zeros((n, 384), dtype=np.float32)
    offset = 7 if config and config.mode != EmbeddingMode.BASE else 0
    for idx in range(n):
        arr[idx, (idx + offset) % 384] = 1.0
    return arr


def test_ablation_returns_base_enriched_sections(monkeypatch):
    monkeypatch.setattr(ablation, "generate_domain_embeddings", fake_embeddings)
    monkeypatch.setattr(
        ablation,
        "reduce_embeddings",
        lambda embeddings, random_state=42: (embeddings[:, :10], embeddings[:, :2]),
    )
    monkeypatch.setattr(
        ablation,
        "cluster_requirements",
        lambda embeddings_10d, min_cluster_size=None, min_samples=3: (
            np.array([0 if i < embeddings_10d.shape[0] // 2 else 1 for i in range(embeddings_10d.shape[0])]),
            np.ones(embeddings_10d.shape[0], dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        ablation,
        "label_clusters",
        lambda texts, labels: {
            0: {"label": "First", "keywords": ["first"], "size": int(np.sum(labels == 0))},
            1: {"label": "Second", "keywords": ["second"], "size": int(np.sum(labels == 1))},
        },
    )
    monkeypatch.setattr(
        ablation,
        "build_similarity_graph",
        lambda **kwargs: {"nodes": [{"id": i} for i in range(len(kwargs["texts"]))], "edges": []},
    )

    report = ablation.run_embedding_ablation(
        ["A", "B", "C", "D", "E", "F"],
        ["ctx"] * 6,
        mode=EmbeddingMode.HYBRID,
    )

    json.dumps(report)
    assert report["mode"] == "hybrid"
    assert "base" in report
    assert "enriched" in report
    assert report["base"]["embedding_shape"] == [6, 384]
    assert report["enriched"]["n_clusters"] == 2
    assert not hasattr(ablation, "SessionLocal")


def test_ablation_handles_small_datasets_with_warnings(monkeypatch):
    monkeypatch.setattr(ablation, "generate_domain_embeddings", fake_embeddings)

    report = ablation.run_embedding_ablation(["A", "B"], [None, None])

    assert report["base"]["n_clusters"] == 0
    assert report["enriched"]["noise_count"] == 2
    assert report["warnings"]
