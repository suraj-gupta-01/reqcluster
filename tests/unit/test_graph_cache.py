from pathlib import Path

from reqcluster.config import get_settings
from reqcluster.db import init_db
from reqcluster.domain import ClusteringParams, RequirementIn
from reqcluster.ingest import IngestResult
from reqcluster.services import clustering, sessions


def _session(word):
    reqs = [RequirementIn(req_id=f"R{i}", text=f"The {word} shall report status {i}", ordinal=i)
            for i in range(12)]
    sid = sessions.create(IngestResult(reqs, {}), "t").id
    clustering.start(sid, ClusteringParams())
    return sid


def test_recreated_session_does_not_get_deleted_sessions_graph():
    init_db()
    sid = _session("alpha")
    assert b"alpha" in sessions.graph_json(sid)
    sessions.delete(sid)
    sid2 = _session("bravo")
    assert sid2 == sid  # SQLite reuses the id: the cache key must still differ
    g = sessions.graph_json(sid2)
    assert b"bravo" in g and b"alpha" not in g


def test_unlink_permission_error_is_not_fatal(monkeypatch, caplog):
    init_db()
    get_settings().runs_dir.mkdir(parents=True, exist_ok=True)
    (get_settings().runs_dir / "7.npz").write_bytes(b"x")

    def boom(self, *a, **k):
        raise PermissionError("locked")

    monkeypatch.setattr(Path, "unlink", boom)
    sessions.unlink_runs([7])
    assert "7.npz" in caplog.text
