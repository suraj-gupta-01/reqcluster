"""Tests for the LLM-backed refinement provider and its deterministic fallback."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from llm_services.refinement import (
    MockClusterRefinementProvider,
    LLMClusterRefinementProvider,
    get_refinement_provider,
)
from llm_services import providers


def test_get_provider_mock_and_llm():
    assert isinstance(get_refinement_provider("mock"), MockClusterRefinementProvider)
    assert isinstance(get_refinement_provider("local"), LLMClusterRefinementProvider)
    assert isinstance(get_refinement_provider("openai"), LLMClusterRefinementProvider)


def test_unsupported_provider_raises():
    import pytest

    with pytest.raises(ValueError):
        get_refinement_provider("definitely-not-a-provider")


def test_llm_provider_falls_back_to_template_when_unconfigured(monkeypatch):
    # No LLM configured -> generate_completion raises -> deterministic fallback.
    monkeypatch.delenv("REQCLUSTER_LOCAL_LLM_URL", raising=False)
    provider = LLMClusterRefinementProvider("local")
    text = provider.generate_merge_rationale("Thermal", "Power", 0.8, 0.01, 0.7, 0.6)
    assert isinstance(text, str) and len(text) > 0
    summary = provider.generate_cluster_summary("Thermal", ["temperature", "cooling"], ["The fan shall run."])
    assert "Thermal" in summary


def test_llm_provider_uses_completion_when_available(monkeypatch):
    monkeypatch.setattr(providers, "generate_completion", lambda *a, **k: "LLM-GENERATED RATIONALE")
    provider = LLMClusterRefinementProvider("local")
    text = provider.generate_split_rationale("Avionics", 0.6, 120.0, 0.4, [5, 4])
    assert text == "LLM-GENERATED RATIONALE"


def test_llm_coherence_is_deterministic_numeric(monkeypatch):
    # Coherence must stay deterministic even for the LLM provider.
    monkeypatch.setattr(providers, "generate_completion", lambda *a, **k: "should not be used")
    provider = LLMClusterRefinementProvider("local")
    emb = np.ones((4, 8), dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array([0, 0, 0, 0])
    result = provider.score_coherence(emb, ["a", "b", "c", "d"], labels, 0, ["k"])
    assert 0.0 <= result.coherence_score <= 1.0
