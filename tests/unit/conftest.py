import pytest


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("REQCLUSTER_HOME", str(tmp_path))
    monkeypatch.setenv("REQCLUSTER_EMBEDDING_BACKEND", "mock")
    monkeypatch.setenv("REQCLUSTER_LLM_PROVIDER", "mock")
    monkeypatch.setenv("REQCLUSTER_JOB_WORKERS", "0")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    from reqcluster.config import get_settings

    from reqcluster import db, jobs

    get_settings.cache_clear()
    db.reset()
    jobs.reset_runner()
    yield
    jobs.reset_runner()
    db.reset()
    get_settings.cache_clear()
