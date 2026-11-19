import numpy as np

from core import pipeline
from core.domain_embeddings import EmbeddingMode


TEXTS = ["A", "B", "C", "D"]
REQ_IDS = ["REQ-1", "REQ-2", "REQ-3", "REQ-4"]


def make_embeddings(n, offset=0):
    arr = np.zeros((n, 384), dtype=np.float32)
    for idx in range(n):
        arr[idx, (idx + offset) % 384] = 1.0
    return arr


def patch_pipeline(monkeypatch):
    calls = {"base": 0, "domain": 0, "compare": 0, "ablation": 0}

    def fake_generate(texts, batch_size=64, progress_callback=None, use_cache=True):
        calls["base"] += 1
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return make_embeddings(len(texts))

    def fake_domain(texts, enriched_texts=None, config=None, progress_callback=None, use_cache=True):
        calls["domain"] += 1
        calls["domain_mode"] = config.mode.value
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return make_embeddings(len(texts), offset=5)

    monkeypatch.setattr(pipeline, "generate_embeddings", fake_generate)
    monkeypatch.setattr(pipeline, "generate_domain_embeddings", fake_domain)
    monkeypatch.setattr(
        pipeline,
        "reduce_embeddings",
        lambda embeddings, return_reducers=False: (
            (embeddings[:, :10], embeddings[:, :2], None, None)
            if return_reducers
            else (embeddings[:, :10], embeddings[:, :2])
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "cluster_requirements",
        lambda embeddings_10d, min_cluster_size=None, min_samples=3: (
            np.array([0, 0, 1, 1]),
            np.ones(4, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "label_clusters",
        lambda texts, labels: {
            0: {"label": "Alpha", "keywords": ["alpha"], "size": 2},
            1: {"label": "Beta", "keywords": ["beta"], "size": 2},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "build_similarity_graph",
        lambda **kwargs: {"nodes": [{"id": i} for i in range(len(kwargs["texts"]))], "edges": []},
    )

    def fake_compare(base_embeddings, candidate_embeddings):
        calls["compare"] += 1
        return {"n_requirements": len(base_embeddings), "warnings": []}

    def fake_ablation(**kwargs):
        calls["ablation"] += 1
        mode = kwargs["mode"]
        return {"mode": mode.value if isinstance(mode, EmbeddingMode) else str(mode)}

    monkeypatch.setattr(pipeline, "compare_embeddings", fake_compare)
    monkeypatch.setattr(pipeline, "run_embedding_ablation", fake_ablation)
    return calls


def test_default_pipeline_still_uses_base_mode(monkeypatch):
    calls = patch_pipeline(monkeypatch)

    result = pipeline.run_pipeline(TEXTS, REQ_IDS)

    assert result["embedding_mode"] == "base"
    assert result["n_clusters"] == 2
    assert calls["base"] == 1
    assert calls["domain"] == 0
    assert "embedding_comparison" not in result
    assert "ablation_report" not in result


def test_hybrid_mode_routes_through_domain_embedding_builder(monkeypatch):
    calls = patch_pipeline(monkeypatch)

    result = pipeline.run_pipeline(
        TEXTS,
        REQ_IDS,
        embedding_mode="hybrid",
        enriched_texts=["ctx"] * 4,
    )

    assert result["embedding_mode"] == "hybrid"
    assert calls["domain"] == 1
    assert calls["domain_mode"] == "hybrid"


def test_comparison_report_is_included_only_when_requested(monkeypatch):
    calls = patch_pipeline(monkeypatch)

    without = pipeline.run_pipeline(TEXTS, REQ_IDS)
    with_report = pipeline.run_pipeline(
        TEXTS,
        REQ_IDS,
        enable_embedding_comparison=True,
    )

    assert "embedding_comparison" not in without
    assert with_report["embedding_comparison"]["n_requirements"] == 4
    assert calls["compare"] == 1


def test_ablation_report_is_included_only_when_requested(monkeypatch):
    calls = patch_pipeline(monkeypatch)

    without = pipeline.run_pipeline(TEXTS, REQ_IDS)
    with_report = pipeline.run_pipeline(TEXTS, REQ_IDS, run_ablation=True)

    assert "ablation_report" not in without
    assert with_report["ablation_report"]["mode"] == "base"
    assert calls["ablation"] == 1
