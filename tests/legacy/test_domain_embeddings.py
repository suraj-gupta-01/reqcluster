import hashlib

import numpy as np
import pytest

from core import domain_embeddings as de
from core.domain_embeddings import (
    DomainEmbeddingConfig,
    EmbeddingMode,
    build_embedding_texts,
    compute_domain_cache_key,
    generate_domain_embeddings,
)


class FakeModel:
    def __init__(self):
        self.calls = 0

    def encode(self, batch, **kwargs):
        self.calls += 1
        vectors = []
        for text in batch:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            raw = (digest * 12)[: de.EMBEDDING_DIM]
            vector = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) + 1.0
            if kwargs.get("normalize_embeddings", False):
                vector = vector / np.linalg.norm(vector)
            vectors.append(vector)
        return np.vstack(vectors).astype(np.float32)


def test_base_mode_returns_original_texts():
    config = DomainEmbeddingConfig(mode=EmbeddingMode.BASE)
    assert build_embedding_texts(["REQ A", "REQ B"], ["ignored", None], config) == [
        "REQ A",
        "REQ B",
    ]


def test_enriched_mode_returns_enriched_texts():
    config = DomainEmbeddingConfig(mode=EmbeddingMode.ENRICHED)
    assert build_embedding_texts(["REQ A"], ["Expanded domain context"], config) == [
        "Expanded domain context"
    ]


def test_enriched_mode_missing_falls_back_only_when_allowed():
    fallback_config = DomainEmbeddingConfig(
        mode=EmbeddingMode.ENRICHED,
        fallback_to_base=True,
    )
    assert build_embedding_texts(["REQ A"], [None], fallback_config) == ["REQ A"]

    strict_config = DomainEmbeddingConfig(
        mode=EmbeddingMode.ENRICHED,
        fallback_to_base=False,
    )
    with pytest.raises(ValueError, match="Missing enriched text"):
        build_embedding_texts(["REQ A"], [None], strict_config)


def test_hybrid_mode_uses_deterministic_section_markers():
    config = DomainEmbeddingConfig(mode=EmbeddingMode.HYBRID)
    assert build_embedding_texts(["Original"], ["Context"], config) == [
        "Original Requirement:\nOriginal\n\nDomain-Aware Context:\nContext"
    ]


def test_text_length_is_bounded_by_max_text_chars():
    config = DomainEmbeddingConfig(mode=EmbeddingMode.BASE, max_text_chars=5)
    assert build_embedding_texts(["abcdef"], None, config) == ["abcde"]


def test_base_and_hybrid_modes_generate_different_cache_keys():
    base_config = DomainEmbeddingConfig(mode=EmbeddingMode.BASE)
    hybrid_config = DomainEmbeddingConfig(mode=EmbeddingMode.HYBRID)
    base_texts = build_embedding_texts(["A"], ["B"], base_config)
    hybrid_texts = build_embedding_texts(["A"], ["B"], hybrid_config)

    assert compute_domain_cache_key(base_texts, base_config) != compute_domain_cache_key(
        hybrid_texts,
        hybrid_config,
    )


def test_cache_round_trip_uses_existing_valid_cache(tmp_path, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(de, "get_model", lambda: model)

    config = DomainEmbeddingConfig(mode=EmbeddingMode.BASE)
    first = generate_domain_embeddings(["A", "B"], config=config)

    class RaisingModel:
        def encode(self, *args, **kwargs):
            raise AssertionError("cache was not used")

    monkeypatch.setattr(de, "get_model", lambda: RaisingModel())
    second = generate_domain_embeddings(["A", "B"], config=config)

    assert model.calls == 1
    np.testing.assert_allclose(first, second)


def test_corrupt_cache_file_regenerates_safely(tmp_path, monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(de, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(de, "get_model", lambda: model)

    config = DomainEmbeddingConfig(mode=EmbeddingMode.HYBRID)
    texts = build_embedding_texts(["A"], ["B"], config)
    cache_file = de._cache_path(compute_domain_cache_key(texts, config))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_bytes(b"not a valid numpy file")

    embeddings = generate_domain_embeddings(["A"], ["B"], config=config)

    assert model.calls == 1
    assert embeddings.shape == (1, de.EMBEDDING_DIM)
    assert embeddings.dtype == np.float32


def test_cache_write_is_atomic(tmp_path, monkeypatch):
    model = FakeModel()
    replace_calls = []
    original_replace = de.os.replace

    def recording_replace(src, dst):
        replace_calls.append((src, dst))
        return original_replace(src, dst)

    monkeypatch.setattr(de, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(de, "get_model", lambda: model)
    monkeypatch.setattr(de.os, "replace", recording_replace)

    generate_domain_embeddings(["A"], config=DomainEmbeddingConfig(mode=EmbeddingMode.BASE))

    assert replace_calls
    assert len(list(tmp_path.glob("domain_embeddings_*.npy"))) == 1
    assert not list(tmp_path.glob("*.tmp"))
