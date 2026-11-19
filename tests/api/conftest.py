import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("REQCLUSTER_HOME", str(tmp_path))
    monkeypatch.setenv("REQCLUSTER_EMBEDDING_BACKEND", "mock")
    monkeypatch.setenv("REQCLUSTER_LLM_PROVIDER", "mock")
    monkeypatch.setenv("REQCLUSTER_JOB_WORKERS", "0")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from reqcluster import db, jobs
    from reqcluster.app import create_app
    from reqcluster.config import get_settings

    def reset():
        get_settings.cache_clear()
        jobs.reset_runner()
        db.reset()

    reset()
    with TestClient(create_app()) as c:
        yield c
    reset()
