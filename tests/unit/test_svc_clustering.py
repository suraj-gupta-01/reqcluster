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
    with pytest.raises(errors.Conflict):  # enriched mode needs a finished enrichment first
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


def test_hybrid_mode_needs_enrichment_then_embeds_enriched_text(monkeypatch):
    from reqcluster import embeddings
    from reqcluster.services import enrichment

    sid = make(12)
    with pytest.raises(errors.Conflict):
        clustering.start(sid, ClusteringParams(embedding_mode="hybrid"))
    assert enrichment.start(sid, "mock").status == "succeeded"
    seen = []
    real = embeddings.embed
    monkeypatch.setattr(embeddings, "embed", lambda t, progress=None: seen.extend(t) or real(t))
    assert clustering.start(sid, ClusteringParams(embedding_mode="hybrid")).status == "succeeded"
    assert seen and all("Domain-Aware Context" in t for t in seen)
    labels = [c.label for c in sessions.clusters(sid)]
    assert not any("Domain-Aware" in lab or "Original" in lab for lab in labels)


def test_failed_persist_leaves_no_orphan_npz(monkeypatch):
    from reqcluster.config import get_settings

    sid = make()
    clustering.start(sid, ClusteringParams())  # run 1 succeeds
    before = sorted(p.name for p in get_settings().runs_dir.glob("*.npz"))
    monkeypatch.setattr(clustering, "KEEP_RUNS", "boom")  # prune step fails after the npz write
    assert clustering.start(sid, ClusteringParams(seed=7)).status == "failed"
    assert sorted(p.name for p in get_settings().runs_dir.glob("*.npz")) == before


def test_search_matches_req_id_too():
    sid = make(texts=["The pump shall start", "The valve shall close", "The fan shall spin"])
    items, total = sessions.requirements(sid, search="R1")
    assert total == 1 and items[0].req_id == "R1"
