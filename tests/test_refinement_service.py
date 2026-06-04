"""Tests for Phase 3 refinement service layer."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from core.representatives import (
    Representative,
    extract_representatives,
    extract_cluster_summary_texts,
)
from llm_services.refinement import (
    MockClusterRefinementProvider,
    get_refinement_provider,
    score_all_clusters,
    CoherenceResult,
)


def _make_clustered_data(n_clusters=3, n_per_cluster=10, dim=384):
    np.random.seed(42)
    embeddings = []
    labels = []
    texts = []
    req_ids = []
    for i in range(n_clusters):
        center = np.random.randn(dim)
        center = center / np.linalg.norm(center)
        points = center + np.random.randn(n_per_cluster, dim) * 0.05
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        points = points / norms
        embeddings.append(points)
        for j in range(n_per_cluster):
            labels.append(i)
            texts.append(f"Requirement {i}-{j} for domain {i}")
            req_ids.append(f"REQ-{i * n_per_cluster + j + 1:03d}")
    return np.vstack(embeddings), np.array(labels, dtype=np.int32), texts, req_ids


class TestExtractRepresentatives:
    def test_returns_reps_per_cluster(self):
        emb, labels, texts, req_ids = _make_clustered_data(3, 10)
        reps = extract_representatives(emb, texts, labels, req_ids, top_n=3)
        assert set(reps.keys()) == {0, 1, 2}
        for cid, rep_list in reps.items():
            assert len(rep_list) <= 3
            for r in rep_list:
                assert isinstance(r, Representative)
                assert r.similarity_to_centroid > 0
                assert r.rank >= 1

    def test_excludes_noise(self):
        emb, labels, texts, req_ids = _make_clustered_data(2, 10)
        labels[0] = -1
        reps = extract_representatives(emb, texts, labels, req_ids)
        assert -1 not in reps

    def test_single_point_cluster(self):
        emb = np.random.randn(1, 384).astype(np.float32)
        labels = np.array([0], dtype=np.int32)
        texts = ["Single requirement"]
        req_ids = ["REQ-001"]
        reps = extract_representatives(emb, texts, labels, req_ids, top_n=3)
        assert len(reps[0]) == 1


class TestExtractClusterSummaryTexts:
    def test_returns_summaries(self):
        emb, labels, texts, req_ids = _make_clustered_data(2, 10)
        summaries = extract_cluster_summary_texts(emb, texts, labels, req_ids, top_n=2)
        assert set(summaries.keys()) == {0, 1}
        for summary in summaries.values():
            assert isinstance(summary, str)
            assert len(summary) > 0


class TestMockClusterRefinementProvider:
    def test_score_coherence(self):
        emb, labels, texts, req_ids = _make_clustered_data(2, 10)
        provider = MockClusterRefinementProvider()
        result = provider.score_coherence(emb, texts, labels, 0, ["keyword1", "keyword2"])
        assert isinstance(result, CoherenceResult)
        assert 0 <= result.coherence_score <= 1
        assert result.cluster_id == 0
        assert len(result.assessment) > 0

    def test_generate_merge_rationale(self):
        provider = MockClusterRefinementProvider()
        rationale = provider.generate_merge_rationale(
            "Thermal Control", "Power Systems", 0.85, 0.02, 0.9, 0.8
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 20
        assert "Thermal Control" in rationale
        assert "Power Systems" in rationale

    def test_generate_split_rationale(self):
        provider = MockClusterRefinementProvider()
        rationale = provider.generate_split_rationale(
            "Mixed Requirements", 0.7, 150.0, 0.35, [12, 8]
        )
        assert isinstance(rationale, str)
        assert len(rationale) > 20

    def test_generate_cluster_summary(self):
        provider = MockClusterRefinementProvider()
        summary = provider.generate_cluster_summary(
            "Thermal Control",
            ["thermal", "temperature", "cooling"],
            ["The system shall maintain temperature", "Cooling system shall operate"],
        )
        assert isinstance(summary, str)
        assert "Thermal Control" in summary


class TestGetRefinementProvider:
    def test_mock_provider(self):
        provider = get_refinement_provider("mock")
        assert provider.name == "mock"

    def test_unsupported_raises(self):
        with pytest.raises(ValueError):
            get_refinement_provider("unsupported_provider")


class TestScoreAllClusters:
    def test_scores_all(self):
        emb, labels, texts, _ = _make_clustered_data(3, 10)
        cluster_info = {
            0: {"label": "Cluster 0", "keywords": ["kw0"], "size": 10},
            1: {"label": "Cluster 1", "keywords": ["kw1"], "size": 10},
            2: {"label": "Cluster 2", "keywords": ["kw2"], "size": 10},
        }
        results = score_all_clusters(emb, texts, labels, cluster_info, "mock")
        assert set(results.keys()) == {0, 1, 2}
        for cr in results.values():
            assert isinstance(cr, CoherenceResult)
