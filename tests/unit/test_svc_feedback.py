import pytest
from sqlalchemy import select

from reqcluster import errors
from reqcluster.db import session_scope
from reqcluster.domain import ClusteringParams
from reqcluster.models import JobRow, RequirementRow
from reqcluster.services import clustering, feedback, sessions
from tests.unit.test_svc_clustering import make


def clustered(n=30):
    sid = make(n)
    clustering.start(sid, ClusteringParams())
    return sid


def labels(sid):
    items, _ = sessions.requirements(sid)
    return {i.id: i.cluster_id for i in items}


def two_clusters(sid):
    lab = labels(sid)
    rid = next(iter(lab))
    other = next(c for c in lab.values() if c not in (lab[rid], -1))
    return rid, lab[rid], other


def test_move_creates_feedback_and_constraints():
    sid = clustered()
    rid, src, dst = two_clusters(sid)
    out = feedback.submit(sid, rid, dst, "because", "me")
    assert out.status == "pending" and out.old_cluster_id == src and out.constraints_added > 0
    assert labels(sid)[rid] == dst
    assert [f.id for f in feedback.queue(sid, "pending")] == [out.feedback_id]
    pairs = feedback.constraints(sid).constraint_pairs
    assert pairs and all(p.a < p.b for p in pairs)
    assert feedback.queue(sid, "approved") == []
    q = feedback.queue(sid, "pending")[0]
    assert q.text and q.req_id


def test_noop_null_and_minus_one():
    sid = clustered()
    rid, src, _ = two_clusters(sid)
    assert feedback.submit(sid, rid, src).feedback_id is None
    a = feedback.submit(sid, rid, None)
    b = feedback.submit(sid, rid, -1)
    assert a.new_cluster_id == -1 and b.feedback_id is None and labels(sid)[rid] == -1
    feedback.submit(sid, rid, src)
    items, _ = sessions.requirements(sid)
    assert next(i for i in items if i.id == rid).is_noise is False


def test_reject_reverts_and_deletes_constraints():
    sid = clustered()
    rid, src, dst = two_clusters(sid)
    fid = feedback.submit(sid, rid, dst).feedback_id
    out = feedback.review(sid, fid, False)
    assert out.status == "rejected" and labels(sid)[rid] == src
    assert feedback.constraints(sid).constraint_pairs == []
    assert feedback.review(sid, feedback.submit(sid, rid, dst).feedback_id, True).status == "approved"


def test_errors():
    sid = clustered()
    rid, _, _ = two_clusters(sid)
    with pytest.raises(errors.NotFound):
        feedback.submit(sid, 99999, -1)
    with pytest.raises(errors.NotFound):
        feedback.submit(sid, rid, 4242)
    with pytest.raises(errors.NotFound):
        feedback.review(sid, 99999, True)
    with pytest.raises(errors.NotFound):
        feedback.submit(999, rid, -1)
    fresh = make(5)
    with session_scope() as s:
        r0 = s.scalar(select(RequirementRow.id).where(RequirementRow.session_id == fresh))
    with pytest.raises(errors.Conflict):
        feedback.submit(fresh, r0, -1)
    with session_scope() as s:
        s.add(JobRow(id="j1", session_id=sid, kind="cluster", status="running", params={}))
    with pytest.raises(errors.Conflict):
        feedback.submit(sid, rid, -1)


def test_constraints_dropped_by_plain_cluster_kept_by_recluster():
    sid = clustered()
    rid, _, dst = two_clusters(sid)
    feedback.submit(sid, rid, dst)
    n = len(feedback.constraints(sid).constraint_pairs)
    job = feedback.recluster(sid)
    assert job.status == "succeeded", job
    assert sessions.get(sid).revision == 2
    assert len(feedback.constraints(sid).constraint_pairs) == n
    assert labels(sid)[rid] == dst
    clustering.start(sid, ClusteringParams())
    assert feedback.constraints(sid).constraint_pairs == [] and feedback.queue(sid) == []


def test_export_guards_formulas():
    sid = clustered()
    rid, _, _ = two_clusters(sid)
    feedback.submit(sid, rid, -1, comments="=1+1", applied_by="@evil")
    body, media = feedback.export(sid, "csv")
    text = body.decode()
    assert media == "text/csv" and "'=1+1" in text and "'@evil" in text
    assert ",=1+1" not in text


def test_confidence():
    assert feedback.confidence("", "x", []) == 1.0
    assert feedback.confidence("about brake pedal", "brake pedal", ["brake"]) > 0.5
    assert feedback.confidence("zzz", "brake", ["pedal"]) == 0.5



def test_review_requires_idle_and_pending(monkeypatch):
    sid = clustered()
    rid, src, dst = two_clusters(sid)
    fid = feedback.submit(sid, rid, dst).feedback_id
    runner = feedback.get_runner()
    real = runner.latest
    monkeypatch.setattr(runner, "latest", lambda *a, **k: type("J", (), {"status": "running", "kind": "recluster"})())
    with pytest.raises(errors.Conflict):
        feedback.review(sid, fid, False)
    monkeypatch.setattr(runner, "latest", real)
    feedback.review(sid, fid, False)
    with pytest.raises(errors.Conflict):
        feedback.review(sid, fid, True)
    assert labels(sid)[rid] == src


def test_recluster_reports_moved():
    sid = clustered()
    rid, _, dst = two_clusters(sid)
    feedback.submit(sid, rid, dst)
    job = feedback.recluster(sid)
    assert job.result["moved"] >= 0


def test_list_sessions_batches_job_lookup(monkeypatch):
    sid = clustered()
    monkeypatch.setattr(sessions.get_runner(), "latest",
                        lambda *a: pytest.fail("per-session latest() query"))
    out = [s for s in sessions.list_sessions() if s.id == sid][0]
    assert out.status == "done" and out.last_job.status == "succeeded"


def test_graph_cache_invalidated_by_edit():
    sid = clustered()
    g = sessions.graph(sid)
    assert sessions.graph(sid) is g and len(g.nodes.id) == 30 and g.nodes.label[0]
    rid, _, dst = two_clusters(sid)
    feedback.submit(sid, rid, dst, "x", "me")
    g2 = sessions.graph(sid)
    assert g2 is not g and g2.edit_seq == g.edit_seq + 1
    assert g2.nodes.cluster_id[g2.nodes.id.index(rid)] == dst


def test_pending_feedback_stays_reviewable_after_recluster():
    # recluster carried its constraints but left the row on the old run: unreviewable, unrejectable
    sid = clustered()
    rid, src, dst = two_clusters(sid)
    fb = feedback.submit(sid, rid, dst, "", "me")
    assert feedback.recluster(sid).status == "succeeded"
    assert [f.id for f in feedback.queue(sid, "pending")] == [fb.feedback_id]
    feedback.review(sid, fb.feedback_id, False)
    assert not [p for p in feedback.constraints(sid).constraint_pairs]


def test_move_to_noise_stores_zero_membership():
    sid = clustered()
    rid = next(iter(labels(sid)))
    feedback.submit(sid, rid, None, "", "me")
    item = next(i for i in sessions.requirements(sid)[0] if i.id == rid)
    assert item.is_noise and item.probability == 0.0
