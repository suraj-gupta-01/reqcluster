import numpy as np
import pytest

from reqcluster import embeddings as E
from reqcluster.errors import Unavailable
from reqcluster.texts import build_texts


class Fake:
    name, revision, is_mock, dim = "fake", "1", False, 8

    def __init__(self):
        self.seen: list[str] = []

    def encode(self, texts, batch_size):
        self.seen += texts
        out = np.array([[len(t) + j for j in range(8)] for t in texts], dtype=np.float32)
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def test_get_embedder_cached():
    assert E.get_embedder() is E.get_embedder()


def test_mock_deterministic_normalised():
    m = E.HashingMockEmbedder()
    a, b = m.encode(["Hello world", "hello  WORLD"], 8)
    assert np.allclose(a, b)
    assert np.isclose(np.linalg.norm(a), 1.0, atol=1e-5)
    assert not m.encode([""], 8).any()


def test_embed_dedupes_and_caches(monkeypatch):
    fake = Fake()
    monkeypatch.setattr(E, "get_embedder", lambda: fake)
    calls = []
    out = E.embed(["a", "bb", "a"], lambda d, t: calls.append((d, t)))
    assert out.shape == (3, 8) and out.dtype == np.float32
    assert np.array_equal(out[0], out[2])
    assert sorted(fake.seen) == ["a", "bb"]
    assert calls[-1] == (2, 2)
    fake.seen.clear()
    out2 = E.embed(["bb", "a"])
    assert fake.seen == []
    assert np.allclose(out2[0], out[1])


def test_mock_never_stored(tmp_path):
    out = E.embed(["x y", "z"])
    assert out.shape == (2, 384)
    assert E.embed([]).shape == (0, 384)
    assert not (tmp_path / "cache" / "embeddings.sqlite").exists()


def test_sbert_load_failure(monkeypatch):
    import sentence_transformers

    def boom(*a, **k):
        raise OSError("no net")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", boom)
    with pytest.raises(Unavailable, match="could not be loaded"):
        E.SbertEmbedder("m", "r", "cpu")


def test_build_texts_modes():
    assert build_texts(["A", "B"], "base", ["x", None]) == (["A", "B"], [])
    assert build_texts(["A"], "enriched", ["Ctx"]) == (["Ctx"], [])
    texts, warns = build_texts(["A"], "enriched", [None])
    assert texts == ["A"] and warns
    assert build_texts(["O"], "hybrid", ["C"])[0] == [
        "Original Requirement:\nO\n\nDomain-Aware Context:\nC"
    ]
    assert build_texts(["a\r\nb"], "base")[0] == ["a\nb"]
    with pytest.raises(ValueError):
        build_texts(["A"], "hybrid", [])


def test_sbert_always_loads_float32(monkeypatch):
    # gte-modernbert's config defaults to fp16: on CPU that ran 3.6 texts/s vs 17.3 in float32
    import sentence_transformers
    import torch

    seen = {}

    class ST:
        def __init__(self, name, **kw):
            seen.update(kw)

        def get_sentence_embedding_dimension(self):
            return 8

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", ST)
    E.SbertEmbedder("Alibaba-NLP/gte-modernbert-base", "1", "cpu")
    assert seen["model_kwargs"]["dtype"] is torch.float32 and seen["device"] == "cpu"


@pytest.mark.parametrize("cuda,expected", [(True, "Alibaba-NLP/gte-modernbert-base"),
                                           (False, "all-MiniLM-L6-v2")])
def test_auto_model_follows_device(monkeypatch, cuda, expected):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    assert E.resolve_model("auto", "auto") == (expected, "cuda" if cuda else "cpu")
    assert E.resolve_model("auto", "cpu") == ("all-MiniLM-L6-v2", "cpu")  # forced CPU stays fast
    assert E.resolve_model("my/model", "auto")[0] == "my/model"  # explicit choice wins


def test_sbert_passes_revision_to_loader(monkeypatch):
    # the configured revision keyed the cache but never reached the loader, so it pinned nothing
    import sentence_transformers

    seen = {}

    class Fake:
        def __init__(self, name, **kw):
            seen.update(kw)

        def get_embedding_dimension(self):
            return 4

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", Fake)
    E.SbertEmbedder("m", "abc123", "cpu")
    assert seen["revision"] == "abc123"
