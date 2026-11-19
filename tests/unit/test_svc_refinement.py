import pytest
from sqlalchemy import select

from reqcluster import errors
from reqcluster.db import session_scope
from reqcluster.domain import ClusteringParams
from reqcluster.models import AuditRow
from reqcluster.services import clustering, refinement, sessions

from .test_svc_clustering import make

LOOSE = refinement.SuggestParams(sim_threshold=0.0, spread_threshold=0.0)


def setup(n=60):
    sid = make(n)
    clustering.start(sid, ClusteringParams(min_cluster_size=5))
    return sid


def gen(sid, kind=None):
    job = refinement.start(sid, LOOSE)
    assert job.status == "succeeded", job
    items = refinement.list_suggestions(sid)
    return [i for i in items if kind in (None, i.kind)]


def test_not_clustered_409():
    sid = make(10)
    with pytest.raises(errors.Conflict):
        refinement.start(sid, LOOSE)


def test_generate_and_regenerate_supersedes_pending():
    sid = setup()
    first = gen(sid, "merge")
    assert first and all(i.status == "pending" and not i.stale and i.rationale for i in first)
    gen(sid, "merge")
    pending = {i.id for i in refinement.list_suggestions(sid, "pending")}
    old = {i.id for i in first}
    assert pending and old.isdisjoint(pending)
    assert {i.status for i in refinement.list_suggestions(sid) if i.id in old} == {"rejected"}


def test_accept_merge_reduces_clusters_and_audits():
    sid = setup()
    sug = gen(sid, "merge")[0]
    before = sessions.get(sid).total_clusters
    out = refinement.decide(sid, sug.id, True, "me")
    assert out.status == "accepted"
    assert sessions.get(sid).total_clusters == before - 1
    assert [a.action for a in refinement.audit(sid)] == ["suggestion_merge"]
    # a decided suggestion cannot be decided again
    with pytest.raises(errors.Conflict):
        refinement.decide(sid, sug.id, False)


def test_accept_split_adds_cluster():
    sid = setup()
    splits = gen(sid, "split")
    if not splits:
        pytest.skip("no bimodal cluster in toy data")
    before = sessions.get(sid).total_clusters
    refinement.decide(sid, splits[0].id, True)
    assert sessions.get(sid).total_clusters == before + 1


def test_reject_audits_without_changing_clusters():
    sid = setup()
    sug = gen(sid)[0]
    before = sessions.get(sid).total_clusters
    assert refinement.decide(sid, sug.id, False, "me").status == "rejected"
    assert sessions.get(sid).total_clusters == before
    assert refinement.audit(sid)[0].action == "suggestion_reject"
    with session_scope() as s:
        assert s.scalars(select(AuditRow)).one().actor == "me"


def test_stale_after_other_accept():
    sid = setup()
    merges = gen(sid, "merge")
    assert len(merges) >= 2
    refinement.decide(sid, merges[0].id, True)
    assert all(i.stale for i in refinement.list_suggestions(sid, "pending"))
    with pytest.raises(errors.Stale):
        refinement.decide(sid, merges[1].id, True)


def test_stale_after_recluster_and_unknown():
    sid = setup()
    sug = gen(sid)[0]
    clustering.start(sid, ClusteringParams(min_cluster_size=3))
    assert refinement.list_suggestions(sid) == []
    with pytest.raises(errors.Stale):
        refinement.decide(sid, sug.id, True)
    with pytest.raises(errors.NotFound):
        refinement.decide(sid, 999999, True)
