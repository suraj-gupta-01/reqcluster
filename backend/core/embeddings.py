import numpy as np
import os
import hashlib
import json
from typing import List, Optional, Callable
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "embeddings")
os.makedirs(CACHE_DIR, exist_ok=True)

_model: Optional[SentenceTransformer] = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


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
    batch_size: int = 64,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Generate SBERT embeddings for a list of texts.
    Returns numpy array of shape (N, 384).
    """
    cache_key = _cache_key(texts)
    cache_file = _cache_path(cache_key)

    if use_cache and os.path.exists(cache_file):
        embeddings = np.load(cache_file)
        if embeddings.shape[0] == len(texts):
            if progress_callback:
                progress_callback(len(texts), len(texts))
            return embeddings

    model = get_model()
    all_embeddings = []
    total = len(texts)

    for start in range(0, total, batch_size):
        batch = texts[start : start + batch_size]
        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        all_embeddings.append(batch_embeddings)
        if progress_callback:
            progress_callback(min(start + batch_size, total), total)

    embeddings = np.vstack(all_embeddings)

    if use_cache:
        np.save(cache_file, embeddings)

    return embeddings
