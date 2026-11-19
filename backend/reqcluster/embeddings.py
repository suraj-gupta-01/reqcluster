"""Embedding backends, text building and cached embedding."""

import functools
import logging
import unicodedata
import zlib
from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from reqcluster.config import get_settings
from reqcluster.errors import Unavailable
from reqcluster.store import EmbeddingStore, key

log = logging.getLogger(__name__)
MAX_TEXT_CHARS = 12_000


class Embedder(Protocol):
    name: str
    revision: str
    is_mock: bool
    dim: int

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray: ...


def adaptive_batch_size(n_texts: int, vram_mb: int | None = None) -> int:
    """Batch size by workload (CPU) or by VRAM when on CUDA."""
    if vram_mb:
        for floor, size in ((12000, 512), (6000, 256), (3000, 128)):
            if vram_mb >= floor:
                return size
        return 64
    if n_texts <= 500:
        return 64
    if n_texts <= 5_000:
        return 128
    if n_texts <= 20_000:
        return 256
    return 512


class SbertEmbedder:
    is_mock = False

    def __init__(self, model_name: str, revision: str, device: str):
        self.name, self.revision = model_name, revision
        try:
            import torch
            from sentence_transformers import SentenceTransformer

            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            self.device = device
            self._model = SentenceTransformer(model_name, device=device)
            self.dim = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:
            raise Unavailable(f"embedding model {model_name!r} could not be loaded: {exc}") from exc

    def vram_mb(self) -> int | None:
        if self.device != "cuda":
            return None
        import torch

        return torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)

    def encode(self, texts: list[str], batch_size: int) -> np.ndarray:
        return self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )


class HashingMockEmbedder:
    name, revision, is_mock = "mock", "0", True

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for tok in text.lower().split():
                out[i] += np.random.default_rng(zlib.crc32(tok.encode())).standard_normal(self.dim)
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


@functools.lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == "mock":
        return HashingMockEmbedder()
    return SbertEmbedder(s.embedding_model, s.embedding_revision, s.embedding_device)


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).replace("\r\n", "\n").replace("\r", "\n")


def _is_missing_enrichment(text: object) -> bool:
    return _normalize_text(text).strip() == ""


def build_texts(
    texts: Sequence[str], mode: str, enriched: Sequence[str | None] | None = None
) -> tuple[list[str], list[str]]:
    """Model input per mode; missing enrichment falls back to base with a warning."""
    originals = list(texts)
    warnings: list[str] = []
    if mode == "base":
        enriched = None
    elif enriched is None:
        warnings.append(f"{mode} mode without enriched texts; using base texts")
        enriched = [None] * len(originals)
    elif len(enriched) != len(originals):
        raise ValueError("texts and enriched must have the same length.")

    out: list[str] = []
    missing = 0
    for i, original in enumerate(originals):
        base = _normalize_text(original)
        value = enriched[i] if enriched is not None else None
        if mode == "base":
            payload = base
        elif _is_missing_enrichment(value):
            payload = base
            missing += 1
        elif mode == "enriched":
            payload = _normalize_text(value)
        else:
            payload = f"Original Requirement:\n{base}\n\nDomain-Aware Context:\n{_normalize_text(value)}"
        out.append(payload[:MAX_TEXT_CHARS])
    if missing and enriched is not None:
        warnings.append(f"{missing} requirement(s) missing enriched text; used base text")
    return out, warnings


def embed(texts: list[str], progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    emb = get_embedder()
    if not texts:
        return np.zeros((0, emb.dim), dtype=np.float32)
    unique = list(dict.fromkeys(texts))
    vecs: dict[str, np.ndarray] = {}
    store = keys = None
    if not emb.is_mock:
        store = EmbeddingStore(get_settings().cache_dir / "embeddings.sqlite", emb.dim)
        keys = {t: key(emb.name, emb.revision, t) for t in unique}
        hits = store.get_many(list(keys.values()))
        vecs = {t: hits[k] for t, k in keys.items() if k in hits}
    misses = [t for t in unique if t not in vecs]
    total = len(unique)
    done = total - len(misses)
    if progress and done:
        progress(done, total)
    if misses:
        vram = emb.vram_mb() if hasattr(emb, "vram_mb") else None
        bs = adaptive_batch_size(len(misses), vram)
        for i in range(0, len(misses), bs):
            batch = misses[i : i + bs]
            arr = np.asarray(emb.encode(batch, bs), dtype=np.float32)
            vecs.update(zip(batch, arr, strict=True))
            if store is not None:
                store.put_many([keys[t] for t in batch], arr)
            done += len(batch)
            if progress:
                progress(done, total)
    return np.vstack([vecs[t] for t in texts]).astype(np.float32)
