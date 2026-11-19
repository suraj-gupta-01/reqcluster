from sqlalchemy import select

from reqcluster.db import session_scope
from reqcluster.domain import ClusteringParams
from reqcluster.models import AuditRow, ClusterRow, RunRow, SessionRow
from reqcluster.services import clustering, sessions
from reqcluster.services.assignments import reassign

from .test_svc_clustering import make


def test_reassign_moves_relabels_and_bumps_edit_seq():
    sid = make()
    clustering.start(sid, ClusteringParams(min_cluster_size=5))
    items, _ = sessions.requirements(sid, page_size=1000)
    first = items[0]
    other = next(i.cluster_id for i in items if i.cluster_id not in (first.cluster_id, -1))
    with session_scope() as s:
        run = s.get(RunRow, s.get(SessionRow, sid).current_run_id)
        before = run.edit_seq
        reassign(s, run, {first.id: -1}, action="test", actor="me")
    with session_scope() as s:
        run = s.get(RunRow, s.get(SessionRow, sid).current_run_id)
        assert run.edit_seq == before + 1
        sizes = {c.cluster_id: c.size for c in s.scalars(
            select(ClusterRow).where(ClusterRow.run_id == run.id))}
        assert -1 not in sizes and run.n_clusters == len(sizes)
        assert s.scalars(select(AuditRow).where(AuditRow.run_id == run.id)).one().actor == "me"
    items2, _ = sessions.requirements(sid, page_size=1000)
    moved = next(i for i in items2 if i.id == first.id)
    assert moved.cluster_id == -1 and moved.is_noise
    assert sessions.get(sid).noise_count == sum(i.is_noise for i in items2)
    assert other in sizes
