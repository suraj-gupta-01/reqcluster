"""
SBERT embedding generation for ReqCluster.

Adaptive batch sizing
---------------------
Rather than a hard-coded batch_size=64, we compute a recommended batch size
at runtime based on the number of texts and available system memory:

    ≤ 500 texts    → batch_size = 64   (small datasets; avoid overhead)
    500–5 K texts  → batch_size = 128
    5 K–20 K texts → batch_size = 256
    > 20 K texts   → batch_size = 512  (large scale; maximise GPU/CPU util)

Callers can still pass an explicit batch_size to override this.

Redis per-text cache
--------------------
Before encoding any texts we check the Redis cache (embedding_cache.py).
Only the *uncached* texts are sent to the model.  Results are written back
to Redis after inference.  If Redis is unavailable, the function falls
through to the existing session-level .npy file cache unchanged.

Cache hierarchy (fastest → slowest):
    1. Redis  — per-text, 30-day TTL, shared across sessions
    2. .npy   — per-batch session cache, local disk
    3. SBERT  — full model inference
"""

import logging
import os
import hashlib
from typing import List, Optional, Callable

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    _torch_available = True
except (OSError, ImportError):
    logging.warning(
        "Failed to load sentence_transformers/PyTorch (likely due to DLL error). "
        "Using local fallback mock."
    )
    _torch_available = False

    class SentenceTransformer:
        def __init__(self, model_name: str, **kwargs):
            self.model_name = model_name

        def encode(
            self,
            sentences: List[str],
            batch_size: int = 32,
            show_progress_bar: bool = False,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = True,
            **kwargs,
        ) -> np.ndarray:
            vectors = []
            for text in sentences:
                # Deterministic mock vector seeded from the text's SHA-256.
                digest = hashlib.sha256(str(text).encode("utf-8")).digest()
                seed = int.from_bytes(digest[:8], "big", signed=False)
                rng = np.random.default_rng(seed)
                vector = rng.standard_normal(384).astype(np.float32)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm
                vectors.append(vector)
            return np.vstack(vectors).astype(np.float32)


from .embedding_cache import get_cached_embeddings_batch, set_cached_embeddings_batch

logger = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def compute_adaptive_batch_size(n_texts: int, override: Optional[int] = None) -> int:
    """
    Return an optimal SBERT batch size for *n_texts*.

    Larger batches amortise Python/ONNX/GPU overhead but consume more memory.
    The thresholds below are based on all-MiniLM-L6-v2 at 384-dim output;
    adjust if you switch to a larger model.

    Override the heuristic by passing an explicit *override* value (e.g. from
    a user-facing API parameter).
    """
    if override is not None and override > 0:
        return override
    if n_texts <= 500:
        return 64
    if n_texts <= 5_000:
        return 128
    if n_texts <= 20_000:
        return 256
    return 512


def _cache_key(texts: List[str]) -> str:
    # Hash in the given order: cached embeddings are saved in this same order,
    # so the key must be order-sensitive to avoid returning misaligned vectors
    # for two inputs that share the same texts in a different order.
    joined = "\n".join(texts)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def _cache_path(key: str) -> str:
    return os.path.join(CACHE_DIR, f"embeddings_{key}.npy")


def generate_embeddings(
    texts: List[str],
    batch_size: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Generate SBERT embeddings for a list of texts.
    Returns numpy array of shape (N, 384).

    batch_size
        If None (default), an adaptive size is computed from len(texts).
        Pass an explicit integer to override.
    """
    # ── Layer 1: session-level .npy file cache ───────────────────────────────
    cache_key = _cache_key(texts)
    cache_file = _cache_path(cache_key)

    if use_cache and os.path.exists(cache_file):
        try:
            embeddings = np.load(cache_file, allow_pickle=False, mmap_mode="r")
        except Exception:
            embeddings = None
        if (
            isinstance(embeddings, np.ndarray)
            and embeddings.ndim == 2
            and embeddings.shape[0] == len(texts)
            and np.all(np.isfinite(embeddings))
        ):
            if progress_callback:
                progress_callback(len(texts), len(texts))
            return embeddings

    # ── Layer 2: per-text Redis cache ────────────────────────────────────────
    # Check which texts already have cached vectors.
    total = len(texts)
    effective_batch = compute_adaptive_batch_size(total, batch_size)

    cached_vectors: list = get_cached_embeddings_batch(texts)  # list[ndarray | None]
    miss_indices = [i for i, v in enumerate(cached_vectors) if v is None]
    hit_count = total - len(miss_indices)

    if hit_count > 0:
        logger.debug(
            "generate_embeddings: %d/%d texts served from Redis cache", hit_count, total
        )

    if miss_indices:
        # Only encode the texts that were not in Redis
        miss_texts = [texts[i] for i in miss_indices]
        model = get_model()
        miss_embeddings: list[np.ndarray] = []

        for start in range(0, len(miss_texts), effective_batch):
            batch = miss_texts[start: start + effective_batch]
            batch_emb = model.encode(
                batch,
                batch_size=effective_batch,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            miss_embeddings.append(batch_emb)
            if progress_callback:
                encoded_so_far = hit_count + min(start + effective_batch, len(miss_texts))
                progress_callback(encoded_so_far, total)

        miss_matrix = np.vstack(miss_embeddings)

        # Write newly computed embeddings back to Redis
        if use_cache:
            set_cached_embeddings_batch(miss_texts, miss_matrix)

        # Fill results in original order
        for idx, miss_idx in enumerate(miss_indices):
            cached_vectors[miss_idx] = miss_matrix[idx]
    else:
        # All served from cache — fire final progress callback
        if progress_callback:
            progress_callback(total, total)

    embeddings = np.vstack(cached_vectors).astype(np.float32)

    # ── Layer 1 write-back: save full session .npy for ultra-fast re-runs ────
    if use_cache:
        try:
            np.save(cache_file, embeddings)
        except Exception as exc:
            logger.warning("generate_embeddings: could not write .npy cache: %s", exc)

    return embeddings
