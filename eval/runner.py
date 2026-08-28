"""Hermetic execution environment and FastAPI TestClient runner."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Generator, Optional
import numpy as np

# Ensure root and backend are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND_DIR = REPO_ROOT / "backend"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


class MockEncoder:
    """Deterministic dummy sentence encoder for fast smoke checks."""

    def __init__(self, model_name: str = "mock-encoder", **kwargs):
        self.model_name = model_name

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
        **kwargs,
    ) -> np.ndarray:
        vectors = []
        for text in sentences:
            digest = hashlib.sha256(str(text).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "big", signed=False)
            rng = np.random.default_rng(seed)
            vec = rng.standard_normal(384).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


@contextmanager
def hermetic_client(
    model_mode: str = "real",
    provider: str = "mock",
) -> Generator[Any, None, None]:
    """Boot FastAPI inside TestClient with an isolated temp SQLite DB and cache."""
    temp_dir = tempfile.mkdtemp(prefix="reqcluster-eval-")
    temp_path = Path(temp_dir)
    db_path = temp_path / "eval.db"
    cache_path = temp_path / "embeddings"
    cache_path.mkdir(parents=True, exist_ok=True)

    # Save original environment state
    old_env = dict(os.environ)
    os.environ["DATABASE_URL"] = f"sqlite:///{db_path.as_posix()}"
    os.environ["REQCLUSTER_LLM_PROVIDER"] = provider
    os.environ["REQCLUSTER_SQLITE_OK"] = "1"
    os.environ.pop("REDIS_URL", None)

    # Import backend modules after setting DATABASE_URL
    import core.embeddings
    import models.database
    from api import routes
    from main import app
    from fastapi.testclient import TestClient

    # Save original cache dir and model
    orig_cache_dir = core.embeddings.CACHE_DIR
    orig_model = core.embeddings._model

    core.embeddings.CACHE_DIR = str(cache_path)

    if model_mode == "mock":
        core.embeddings._model = MockEncoder()

    # Create tables in the isolated temp database
    models.database.Base.metadata.create_all(bind=models.database.engine)

    try:
        with TestClient(app) as client:
            yield client
    finally:
        # Cleanup monkeypatches & engine
        core.embeddings.CACHE_DIR = orig_cache_dir
        core.embeddings._model = orig_model

        try:
            models.database.engine.dispose()
        except Exception:
            pass

        # Restore environment
        os.environ.clear()
        os.environ.update(old_env)

        # Remove temp directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
