import pytest
from pydantic import ValidationError

from reqcluster.config import get_settings, load_settings
from reqcluster.domain import ClusteringParams


def test_defaults(tmp_path):
    s = get_settings()
    assert s.home == tmp_path
    assert s.embedding_backend == "mock"
    assert s.job_workers == 0
    assert s.database_url == "sqlite:///" + (tmp_path / "reqcluster.db").as_posix()
    assert s.cache_dir == tmp_path / "cache"
    assert s.cors_origins == ("http://localhost:5173", "http://localhost:3000")
    assert s.disable_spacy is False


def test_env_override(monkeypatch):
    monkeypatch.setenv("REQCLUSTER_PORT", "9000")
    monkeypatch.setenv("REQCLUSTER_DISABLE_SPACY", "yes")
    s = load_settings(dotenv=None)
    assert s.port == 9000 and s.disable_spacy is True


def test_dotenv_does_not_override_env(tmp_path):
    f = tmp_path / ".env"
    f.write_text("REQCLUSTER_PORT=1111\nREQCLUSTER_HOST='0.0.0.0'\n", encoding="utf-8")
    s = load_settings({"REQCLUSTER_PORT": "2222"}, dotenv=f)
    assert s.port == 2222 and s.host == "0.0.0.0"


def test_bad_int_names_var():
    with pytest.raises(ValueError, match="REQCLUSTER_PORT"):
        load_settings({"REQCLUSTER_PORT": "abc"}, dotenv=None)


def test_bad_literal_and_range():
    with pytest.raises(ValueError, match="REQCLUSTER_EMBEDDING_BACKEND"):
        load_settings({"REQCLUSTER_EMBEDDING_BACKEND": "gpu"}, dotenv=None)
    with pytest.raises(ValueError, match="REQCLUSTER_LLM_MAX_RETRIES"):
        load_settings({"REQCLUSTER_LLM_MAX_RETRIES": "9"}, dotenv=None)


def test_params_validation():
    assert ClusteringParams().min_cluster_size is None
    for bad in ({"sim_threshold": 1.5}, {"min_samples": 0}, {"bogus": 1}):
        with pytest.raises(ValidationError):
            ClusteringParams(**bad)
