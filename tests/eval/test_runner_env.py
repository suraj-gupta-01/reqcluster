import os

from eval.runner import hermetic_client
from reqcluster.config import get_settings
from reqcluster.embeddings import get_embedder


def test_baseline_model_pinned_and_restored(monkeypatch):
    monkeypatch.delenv("REQCLUSTER_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("REQCLUSTER_EMBEDDING_DEVICE", raising=False)
    monkeypatch.setenv("REQCLUSTER_EMBEDDING_BACKEND", "mock")
    before = get_embedder()
    with hermetic_client(model_mode="mock"):
        s = get_settings()
        assert s.embedding_model == "all-MiniLM-L6-v2" and s.embedding_device == "cpu"
        assert get_embedder() is not before  # cache cleared on entry
    assert "REQCLUSTER_EMBEDDING_MODEL" not in os.environ
    assert "REQCLUSTER_EMBEDDING_DEVICE" not in os.environ


def test_explicit_model_wins(monkeypatch):
    monkeypatch.setenv("REQCLUSTER_EMBEDDING_MODEL", "other")
    with hermetic_client(model_mode="mock"):
        assert get_settings().embedding_model == "other"
