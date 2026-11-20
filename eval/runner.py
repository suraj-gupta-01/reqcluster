"""Hermetic execution environment: new FastAPI app in a TestClient with an isolated home."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
from collections.abc import Generator

_ENV_KEYS = (
    "REQCLUSTER_HOME",
    "DATABASE_URL",
    "REQCLUSTER_JOB_WORKERS",
    "REQCLUSTER_EMBEDDING_BACKEND",
    "REQCLUSTER_LLM_PROVIDER",
    "REQCLUSTER_EMBEDDING_MODEL",
    "REQCLUSTER_EMBEDDING_DEVICE",
)


def _reset_backend() -> None:
    from reqcluster import db, jobs
    from reqcluster.config import get_settings
    from reqcluster.embeddings import get_embedder
    from reqcluster.llm.gateway import reset_gateway

    get_settings.cache_clear()
    get_embedder.cache_clear()
    reset_gateway()
    db.reset()
    jobs.reset_runner()


@contextmanager
def hermetic_client(
    model_mode: str = "real",
    provider: str = "mock",
    home: Path | None = None,
) -> Generator[Any, None, None]:
    """Boot the app with env pointing at a fresh home; jobs run inline (0 workers)."""
    temp_dir = None if home else tempfile.mkdtemp(prefix="reqcluster-eval-")
    home_path = Path(home) if home else Path(temp_dir)
    home_path.mkdir(parents=True, exist_ok=True)

    saved = {k: os.environ.get(k) for k in _ENV_KEYS}
    os.environ.pop("DATABASE_URL", None)
    os.environ["REQCLUSTER_HOME"] = str(home_path)
    os.environ["REQCLUSTER_JOB_WORKERS"] = "0"
    os.environ["REQCLUSTER_EMBEDDING_BACKEND"] = "mock" if model_mode == "mock" else "torch"
    os.environ["REQCLUSTER_LLM_PROVIDER"] = provider
    # pin the baseline model so results do not depend on the host GPU
    os.environ.setdefault("REQCLUSTER_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    os.environ.setdefault("REQCLUSTER_EMBEDDING_DEVICE", "cpu")

    from fastapi.testclient import TestClient
    from reqcluster.app import create_app

    _reset_backend()
    try:
        with TestClient(create_app()) as client:
            yield client
    finally:
        _reset_backend()
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if temp_dir:
            shutil.rmtree(temp_dir, ignore_errors=True)
