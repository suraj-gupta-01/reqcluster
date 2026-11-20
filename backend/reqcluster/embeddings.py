"""Embedding backends, text building and cached embedding."""

import functools
import logging
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from reqcluster.config import get_settings
from reqcluster.errors import Unavailable
from reqcluster.store import EmbeddingStore, key

log = logging.getLogger(__name__)


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
            # float32 always: some models default to fp16, which runs ~5x slower on CPU
            self._model = SentenceTransformer(model_name, device=device,
                                              model_kwargs={"dtype": torch.float32})
            m = self._model  # get_embedding_dimension in sentence-transformers >= 6 (old name deprecated)
            self.dim = int(getattr(m, "get_embedding_dimension", m.get_sentence_embedding_dimension)())
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


# "auto": the accurate model where it is fast. gte-modernbert vs MiniLM, ARI uav10k .998 vs .958,
# multidomain .742 vs .679; speed CPU 17 vs 150 texts/s, GPU 863 vs 3551 (findings: 05 K2).
FAST_MODEL = "all-MiniLM-L6-v2"
ACCURATE_MODEL = "Alibaba-NLP/gte-modernbert-base"


@dataclass(frozen=True)
class Calibration:
    coherence: tuple[float, float, float]  # "Highly" / "Moderately" / "Low" lower bounds
    graph: float  # similarity-graph edge cosine
    merge: float  # merge-suggestion centroid cosine
    dependency: float  # dependency-candidate cosine


# Cosine scales differ per model (multidomain random-pair median: MiniLM .20, gte .56), so every
# threshold is per model. Measured with docs/redesign/findings/exp/refine_calibration.py on
# uav_fms_2k / multidomain / uav_fms_10k:
#   coherence median  MiniLM gold .48/.39/.47  merged .36/.28/.36  random .24/.20/.24
#                     gte    gold .69/.66/.69  merged .63/.60/.63  random .57/.56/.56
#   MiniLM bands .45/.38/.28 sit at 1.0/.69/.24 of the random->gold median span (.23 -> .45);
#   the same positions on gte's span (.563 -> .681) give .68/.64/.59.
#   graph/dependency/merge: MiniLM .5/.45 are the p98/p97 of random-pair cosine and .7 is the p98
#   of gold-centroid-pair cosine on multidomain; the same percentiles on gte are .71/.69/.90.
CALIBRATION: dict[str, Calibration] = {
    "all-MiniLM-L6-v2": Calibration((0.45, 0.38, 0.28), 0.5, 0.7, 0.45),
    "Alibaba-NLP/gte-modernbert-base": Calibration((0.68, 0.64, 0.59), 0.71, 0.90, 0.69),
}


def calibration(model: str | None = None) -> Calibration:
    """Thresholds for a model (default: the current embedder); unknown models use MiniLM's."""
    model = model or get_embedder().name
    return CALIBRATION.get(model, CALIBRATION["all-MiniLM-L6-v2"])


def resolve_model(model: str, device: str) -> tuple[str, str]:
    """(model, device) with 'auto' resolved: cuda if available, gte on cuda, MiniLM on cpu."""
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
    if model == "auto":
        model = ACCURATE_MODEL if device == "cuda" else FAST_MODEL
    return model, device


@functools.lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    s = get_settings()
    if s.embedding_backend == "mock":
        return HashingMockEmbedder()
    model, device = resolve_model(s.embedding_model, s.embedding_device)
    return SbertEmbedder(model, s.embedding_revision, device)


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
