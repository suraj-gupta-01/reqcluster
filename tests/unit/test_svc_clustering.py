import pytest

from reqcluster import errors
from reqcluster.domain import ClusteringParams, RequirementIn
from reqcluster.ingest import IngestResult
from reqcluster.models import RunRow
from reqcluster.db import init_db, session_scope
from reqcluster.services import clustering, sessions
from sqlalchemy import func, select

TOPICS = ["brake pedal caliper hydraulic", "battery cell charger voltage",
          "display screen pixel backlight"]


def make(n=30, texts=None):
    init_db()
    texts = texts or [f"The {TOPICS[i % 3]} subsystem shall report status code {i} quickly"
                      for i in range(n)]
    reqs = [RequirementIn(req_id=f"R{i}", text=t, ordinal=i) for i, t in enumerate(texts)]
    return sessions.create(IngestResult(reqs, {}), "t").id


def test_two_runs():
    sid = make()
    job = clustering.start(sid, ClusteringParams())
    assert job.status == "succeeded"
    s = sessions.get(sid)
    assert s.status == "done" and s.revision == 1 and s.total_clusters >= 1
    clustering.start(sid, ClusteringParams(seed=7))
    s = sessions.get(sid)
    assert s.revision == 2
    with session_scope() as db:
        assert db.scalar(select(func.count()).select_from(RunRow)) == 2
    items, total = sessions.requirements(sid)
    assert total == 30 and all(i.cluster_id is not None for i in items)


def test_small_n_all_noise():
    sid = make(3)
    clustering.start(sid, ClusteringParams())
    s = sessions.get(sid)
    assert s.status == "done" and s.noise_count == 3 and s.total_clusters == 0


def test_not_found_and_conflict():
    init_db()
    with pytest.raises(errors.NotFound):
        clustering.start(99, ClusteringParams())
    sid = make(5)
    with pytest.raises(errors.Conflict):
        sessions.clusters(sid)
    items, total = sessions.requirements(sid)
    assert total == 5 and items[0].cluster_id is None and items[0].is_noise is None
    with pytest.raises(errors.Invalid):
        clustering.start(sid, ClusteringParams(embedding_mode="enriched"))


def test_search_escaping():
    sid = make(texts=["50% of the time", "a_b value here", "plain text here", r"back\slash path"])
    for q, n in [("%", 1), ("_", 1), ("\\", 1), ("PLAIN", 1)]:
        assert sessions.requirements(sid, search=q)[1] == n


def test_prune_keeps_5():
    sid = make(10)
    for _ in range(7):
        clustering.start(sid, ClusteringParams())
    with session_scope() as db:
        revs = sorted(db.scalars(select(RunRow.revision)))
    assert revs == [3, 4, 5, 6, 7]
    from reqcluster.config import get_settings
    assert len(list(get_settings().runs_dir.glob("*.npz"))) == 5
