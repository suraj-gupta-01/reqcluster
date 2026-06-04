from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np

from .embeddings import CACHE_DIR as PHASE1_CACHE_DIR
from .embeddings import MODEL_NAME, get_model


EMBEDDING_DIM = 384
MAX_BATCH_SIZE = 512
MAX_TEXT_CHARS_LIMIT = 100_000
CACHE_DIR = Path(PHASE1_CACHE_DIR)
SAFE_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class EmbeddingMode(str, Enum):
    BASE = "base"
    ENRICHED = "enriched"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class DomainEmbeddingConfig:
    mode: EmbeddingMode = EmbeddingMode.BASE
    batch_size: int = 64
    normalize: bool = True
    fallback_to_base: bool = True
    cache_namespace: str = "v2"
    max_text_chars: int = 12_000

    def __post_init__(self) -> None:
        try:
            mode = EmbeddingMode(self.mode)
        except ValueError as exc:
            raise ValueError("Invalid embedding mode. Use base, enriched, or hybrid.") from exc
        object.__setattr__(self, "mode", mode)

        if not isinstance(self.batch_size, int) or not (1 <= self.batch_size <= MAX_BATCH_SIZE):
            raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}.")
        if not isinstance(self.normalize, bool):
            raise ValueError("normalize must be a boolean.")
        if not isinstance(self.fallback_to_base, bool):
            raise ValueError("fallback_to_base must be a boolean.")
        if (
            not isinstance(self.cache_namespace, str)
            or not self.cache_namespace
            or len(self.cache_namespace) > 64
        ):
            raise ValueError("cache_namespace must be a non-empty string of at most 64 characters.")
        if (
            not isinstance(self.max_text_chars, int)
            or not (1 <= self.max_text_chars <= MAX_TEXT_CHARS_LIMIT)
        ):
            raise ValueError(
                f"max_text_chars must be between 1 and {MAX_TEXT_CHARS_LIMIT}."
            )


def _normalize_text(text: object) -> str:
    if text is None:
        return ""
    normalized = unicodedata.normalize("NFC", str(text))
    return normalized.replace("\r\n", "\n").replace("\r", "\n")


def _is_missing_enrichment(text: object) -> bool:
    return text is None or _normalize_text(text).strip() == ""


def _validate_enriched_texts(
    original_count: int,
    enriched_texts: Sequence[str | None] | None,
    config: DomainEmbeddingConfig,
) -> list[str | None]:
    if enriched_texts is None:
        if config.mode == EmbeddingMode.BASE or config.fallback_to_base:
            return [None] * original_count
        raise ValueError("Enriched texts are required for this embedding mode.")

    enriched_list = list(enriched_texts)
    if len(enriched_list) != original_count:
        raise ValueError("original_texts and enriched_texts must have the same length.")
    return enriched_list


def build_embedding_texts(
    original_texts: Sequence[str],
    enriched_texts: Sequence[str | None] | None,
    config: DomainEmbeddingConfig,
) -> list[str]:
    """
    Build deterministic model input text for base, enriched, or hybrid embedding modes.

    Enriched content is treated as inert plain text. It is never executed,
    interpolated into code, used as a path, or logged by this module.
    """
    originals = list(original_texts)
    enriched = _validate_enriched_texts(len(originals), enriched_texts, config)

    embedding_texts: list[str] = []
    for idx, original in enumerate(originals):
        original_text = _normalize_text(original)

        if config.mode == EmbeddingMode.BASE:
            payload = original_text
        else:
            enriched_value = enriched[idx]
            if _is_missing_enrichment(enriched_value):
                if not config.fallback_to_base:
                    raise ValueError("Missing enriched text for at least one requirement.")
                payload = original_text
            elif config.mode == EmbeddingMode.ENRICHED:
                payload = _normalize_text(enriched_value)
            else:
                payload = (
                    "Original Requirement:\n"
                    f"{original_text}\n\n"
                    "Domain-Aware Context:\n"
                    f"{_normalize_text(enriched_value)}"
                )

        embedding_texts.append(payload[: config.max_text_chars])

    return embedding_texts


def compute_domain_cache_key(
    embedding_texts: Sequence[str],
    config: DomainEmbeddingConfig,
    model_name: str = MODEL_NAME,
) -> str:
    """
    Compute a stable SHA-256 key for domain-aware embedding cache entries.

    The key is order-sensitive because cached arrays are returned in the same
    row order as the input texts.
    """
    payload = {
        "cache_namespace": config.cache_namespace,
        "fallback_to_base": config.fallback_to_base,
        "max_text_chars": config.max_text_chars,
        "mode": config.mode.value,
        "model_name": model_name,
        "normalize": config.normalize,
        "texts": list(embedding_texts),
    }
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _cache_path(cache_key: str) -> Path:
    if not SAFE_DIGEST_RE.fullmatch(cache_key):
        raise ValueError("Invalid cache key.")
    return CACHE_DIR / f"domain_embeddings_{cache_key}.npy"


def _rows_are_l2_normalized(embeddings: np.ndarray, atol: float = 1e-3) -> bool:
    if embeddings.shape[0] == 0:
        return True
    norms = np.linalg.norm(embeddings, axis=1)
    return bool(np.all(np.isfinite(norms)) and np.allclose(norms, 1.0, atol=atol, rtol=atol))


def _load_cached_embeddings(
    cache_file: Path,
    expected_rows: int,
    normalize: bool,
) -> np.ndarray | None:
    try:
        embeddings = np.load(cache_file, allow_pickle=False)
    except Exception:
        return None

    if not isinstance(embeddings, np.ndarray):
        return None
    if embeddings.shape != (expected_rows, EMBEDDING_DIM):
        return None
    if embeddings.dtype != np.float32:
        return None
    if not np.all(np.isfinite(embeddings)):
        return None
    if normalize and not _rows_are_l2_normalized(embeddings):
        return None
    return embeddings


def _atomic_save_npy(cache_file: Path, embeddings: np.ndarray) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=cache_file.parent,
            prefix=f".{cache_file.stem}.",
            suffix=".tmp",
            delete=False,
        ) as tmp:
            tmp_name = tmp.name
            np.save(tmp, embeddings, allow_pickle=False)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, cache_file)
    except Exception:
        if tmp_name and os.path.exists(tmp_name):
            try:
                os.remove(tmp_name)
            except OSError:
                pass
        raise


def _normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    nonzero = norms[:, 0] > 1e-12
    embeddings = embeddings.copy()
    embeddings[nonzero] = embeddings[nonzero] / norms[nonzero]
    return embeddings


def _encode_texts(
    embedding_texts: list[str],
    config: DomainEmbeddingConfig,
    progress_callback: Optional[Callable[[int, int], None]],
) -> np.ndarray:
    if not embedding_texts:
        if progress_callback:
            progress_callback(0, 0)
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    model = get_model()
    batches: list[np.ndarray] = []
    total = len(embedding_texts)

    for start in range(0, total, config.batch_size):
        batch = embedding_texts[start : start + config.batch_size]
        encoded = model.encode(
            batch,
            batch_size=config.batch_size,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=config.normalize,
        )
        batch_embeddings = np.asarray(encoded, dtype=np.float32)
        if batch_embeddings.ndim == 1:
            batch_embeddings = batch_embeddings.reshape(1, -1)
        if batch_embeddings.shape != (len(batch), EMBEDDING_DIM):
            raise ValueError("Embedding model returned an unexpected output shape.")
        batches.append(batch_embeddings)
        if progress_callback:
            progress_callback(min(start + config.batch_size, total), total)

    embeddings = np.vstack(batches).astype(np.float32, copy=False)
    if not np.all(np.isfinite(embeddings)):
        embeddings = np.nan_to_num(embeddings, nan=0.0, posinf=0.0, neginf=0.0).astype(
            np.float32,
            copy=False,
        )
    if config.normalize:
        embeddings = _normalize_rows(embeddings).astype(np.float32, copy=False)
    return embeddings


def generate_domain_embeddings(
    original_texts: Sequence[str],
    enriched_texts: Sequence[str | None] | None = None,
    config: DomainEmbeddingConfig | None = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_cache: bool = True,
) -> np.ndarray:
    """
    Generate mode-aware SBERT embeddings for base, enriched, or hybrid inputs.
    """
    config = config or DomainEmbeddingConfig()
    embedding_texts = build_embedding_texts(original_texts, enriched_texts, config)

    if not embedding_texts:
        if progress_callback:
            progress_callback(0, 0)
        return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

    cache_key = compute_domain_cache_key(embedding_texts, config)
    cache_file = _cache_path(cache_key)

    if use_cache and cache_file.exists():
        cached = _load_cached_embeddings(cache_file, len(embedding_texts), config.normalize)
        if cached is not None:
            if progress_callback:
                progress_callback(len(embedding_texts), len(embedding_texts))
            return cached

    embeddings = _encode_texts(embedding_texts, config, progress_callback)

    if use_cache:
        _atomic_save_npy(cache_file, embeddings)

    return embeddings
