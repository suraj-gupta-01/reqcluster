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
