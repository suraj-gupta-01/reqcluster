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
