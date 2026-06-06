"""
Redis-backed per-text embedding cache for ReqCluster.

Design
------
Each individual text gets its own cache entry keyed by a content-versioned
hash.  This is more granular than the existing session-level .npy file cache:

    .npy cache  — caches the *entire batch* for a run (fast on re-runs, but a
                  single new/changed text invalidates the whole batch).
    Redis cache — caches *each text independently*, so adding new requirements
                  to an existing session avoids re-embedding unchanged texts.

Cache key format
    reqcluster:emb:v1:<sha256(text)>

Value
    Raw float32 bytes (384 × 4 = 1536 bytes per vector).

TTL
    30 days.  Content-versioned keys mean stale vectors are never returned for
    changed text (the text changes → different hash → new key).

Graceful degradation
    All public functions return None / False when Redis is unavailable.  The
    caller (embeddings.py) falls through to model inference, so the app works
    fine without Redis — you just lose the per-text caching benefit.

Configuration
    Set REDIS_URL in .env:
        REDIS_URL=redis://localhost:6379

    If REDIS_URL is not set, the module logs a one-time info notice and all
    operations become no-ops (returns None).
"""

from __future__ import annotations

import hashlib
import logging
import os
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CACHE_VERSION = "v1"
_KEY_PREFIX = f"reqcluster:emb:{_CACHE_VERSION}:"
_TTL_SECONDS = 30 * 24 * 3600  # 30 days
_EMBEDDING_DIM = 384
_DTYPE = np.float32
_BYTES_PER_VECTOR = _EMBEDDING_DIM * _DTYPE(0).itemsize  # 1536 bytes

# ---------------------------------------------------------------------------
# Redis client (lazy init, singleton)
# ---------------------------------------------------------------------------

_redis_client = None
_redis_unavailable = False  # latched to True on first connection failure


def _get_redis():
    """Return a connected Redis client, or None if Redis is not configured / unavailable."""
    global _redis_client, _redis_unavailable

    if _redis_unavailable:
        return None

    if _redis_client is not None:
        return _redis_client

    redis_url = os.getenv("REDIS_URL", "")
    if not redis_url:
        logger.info(
            "ReqCluster embedding cache: REDIS_URL not set — per-text Redis cache "
            "disabled.  Set REDIS_URL=redis://localhost:6379 to enable it."
        )
        _redis_unavailable = True
        return None

    try:
        import redis  # type: ignore[import]
        client = redis.from_url(
            redis_url,
            decode_responses=False,   # we store raw bytes
            socket_connect_timeout=2,  # fail fast if Redis is down
            socket_timeout=1,
        )
        client.ping()  # validate connectivity eagerly
        _redis_client = client
        logger.info("ReqCluster embedding cache: connected to Redis → %s", redis_url)
        return _redis_client
    except ImportError:
        logger.warning(
            "ReqCluster embedding cache: 'redis' package not installed — "
            "per-text cache disabled.  Run: pip install redis"
        )
        _redis_unavailable = True
        return None
    except Exception as exc:
        logger.warning(
            "ReqCluster embedding cache: could not connect to Redis (%s) — "
            "falling back to file cache only.",
            exc,
        )
        _redis_unavailable = True
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _make_key(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{_KEY_PREFIX}{digest}"


def get_cached_embedding(text: str) -> Optional[np.ndarray]:
    """Return the cached float32 embedding vector for *text*, or None on miss/error."""
    client = _get_redis()
    if client is None:
        return None
    try:
        raw = client.get(_make_key(text))
        if raw is None:
            return None
        if len(raw) != _BYTES_PER_VECTOR:
            logger.debug("Embedding cache: unexpected byte length %d for key, ignoring.", len(raw))
            return None
        return np.frombuffer(raw, dtype=_DTYPE).copy()
    except Exception as exc:
        logger.debug("Embedding cache get error: %s", exc)
        return None


def set_cached_embedding(text: str, vector: np.ndarray) -> bool:
    """Store *vector* in Redis with a 30-day TTL.  Returns True on success."""
    client = _get_redis()
    if client is None:
        return False
    try:
        v = np.asarray(vector, dtype=_DTYPE)
        if v.shape != (_EMBEDDING_DIM,) or not np.all(np.isfinite(v)):
            return False
        client.set(_make_key(text), v.tobytes(), ex=_TTL_SECONDS)
        return True
    except Exception as exc:
        logger.debug("Embedding cache set error: %s", exc)
        return False


def get_cached_embeddings_batch(texts: list[str]) -> list[Optional[np.ndarray]]:
    """
    Batch-fetch cached embeddings for a list of texts.

    Returns a list of the same length as *texts*: each element is either a
    float32 ndarray (cache hit) or None (cache miss).  Uses a single Redis
    pipeline call for efficiency.
    """
    client = _get_redis()
    if client is None:
        return [None] * len(texts)
    try:
        pipe = client.pipeline(transaction=False)
        for text in texts:
            pipe.get(_make_key(text))
        raw_results = pipe.execute()
        out: list[Optional[np.ndarray]] = []
        for raw in raw_results:
            if raw is None or len(raw) != _BYTES_PER_VECTOR:
                out.append(None)
            else:
                out.append(np.frombuffer(raw, dtype=_DTYPE).copy())
        return out
    except Exception as exc:
        logger.debug("Embedding cache batch get error: %s", exc)
        return [None] * len(texts)


def set_cached_embeddings_batch(texts: list[str], vectors: np.ndarray) -> None:
    """
    Batch-store embeddings using a single Redis pipeline call.

    *vectors* must be a float32 array of shape (len(texts), EMBEDDING_DIM).
    Silently skips on any error.
    """
    client = _get_redis()
    if client is None:
        return
    try:
        pipe = client.pipeline(transaction=False)
        for text, vec in zip(texts, vectors):
            v = np.asarray(vec, dtype=_DTYPE)
            if v.shape == (_EMBEDDING_DIM,) and np.all(np.isfinite(v)):
                pipe.set(_make_key(text), v.tobytes(), ex=_TTL_SECONDS)
        pipe.execute()
    except Exception as exc:
        logger.debug("Embedding cache batch set error: %s", exc)
