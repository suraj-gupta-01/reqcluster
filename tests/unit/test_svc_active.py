from sqlalchemy import select
import pytest

from reqcluster import errors
from reqcluster.domain import ClusteringParams
from reqcluster.services import active_learning, clustering
from tests.unit.test_svc_clustering import make


def test_uncertainty_and_quality():
    sid = make(30)
    with pytest.raises(errors.Conflict):
        active_learning.uncertainty(sid)
    assert active_learning.quality(sid) == []
    clustering.start(sid, ClusteringParams())
    clustering.start(sid, ClusteringParams(seed=3))
    top = active_learning.uncertainty(sid, 5)
    assert len(top) == 5
    assert [t.uncertainty for t in top] == sorted((t.uncertainty for t in top), reverse=True)
    q = active_learning.quality(sid)
    assert [p.revision for p in q] == [1, 2] and q[0].kind == "cluster"
    with pytest.raises(errors.NotFound):
        active_learning.quality(999)


def test_moved_requirement_has_prob_one_and_leaves_queue():
    from reqcluster.services import feedback
    from tests.unit.test_svc_feedback import two_clusters
    sid = make(30)
    clustering.start(sid, ClusteringParams())
    before = active_learning.uncertainty(sid, 30)
    pick = next(t for t in before if t.uncertainty > 0)
    _, _, dst = two_clusters(sid)
    feedback.submit(sid, pick.requirement_id, dst)
    after = active_learning.uncertainty(sid, 30)
    assert pick.requirement_id not in {t.requirement_id for t in after}
    from reqcluster.db import session_scope
    from reqcluster.models import AssignmentRow
    with session_scope() as s:
        a = s.scalars(select(AssignmentRow).where(AssignmentRow.requirement_id == pick.requirement_id)).one()
        assert a.probability == 1.0
