import pytest

from reqcluster import errors
from reqcluster.db import init_db
from reqcluster.domain import ClusteringParams, RequirementIn
from reqcluster.export.bundle import build as build_bundle
from reqcluster.ingest import IngestResult
from reqcluster.services import clustering, dependencies, sessions

TOPICS = ["brake pedal caliper hydraulic", "battery cell charger voltage",
          "display screen pixel backlight"]


def make(n=30):
    init_db()
    reqs = [RequirementIn(req_id=f"R{i}", ordinal=i,
                          text=f"The {TOPICS[i % 3]} subsystem shall report status code {i} quickly"
                               + (f", as defined in R{i - 1}" if i else ""))
            for i in range(n)]
    sid = sessions.create(IngestResult(reqs, {}), "t").id
    clustering.start(sid, ClusteringParams())
    return sid


def tree(sid):
    job = dependencies.start(sid, dependencies.DependencyRequest(sim_threshold=0.0))
    assert job.status == "succeeded", job.error
    return job.result


def test_requires_clustering():
    init_db()
    sid = sessions.create(IngestResult([RequirementIn(req_id="R", text="x", ordinal=0)], {}),
                          "t").id
    with pytest.raises(errors.Conflict):
        dependencies.start(sid, dependencies.DependencyRequest())


def test_tree_tied_to_run_and_explain_cache():
    sid = make()
    res = tree(sid)
    assert res["edges"], "expected edges at threshold 0"
    assert dependencies.get(sid).run_id == res["run_id"]
    e = res["edges"][0]
    req = dependencies.ExplainRequest(source=e["source"], target=e["target"])
    assert dependencies.explain(sid, req).cached is False
    assert dependencies.explain(sid, req).cached is True
    forced = dependencies.explain(sid, req.model_copy(update={"force": True}))
    assert forced.cached is False
    assert dependencies.explain(sid, req).cached is True
    with pytest.raises(errors.NotFound):
        dependencies.explain(sid, dependencies.ExplainRequest(source=999999, target=999998))
    b = build_bundle(sid)
    assert len(b.edges) == len(res["edges"]) and b.edges[0][0].startswith("R")
    clustering.start(sid, ClusteringParams(seed=7))
    with pytest.raises(errors.Conflict):
        dependencies.get(sid)
    with pytest.raises(errors.NotFound):
        dependencies.explain(sid, req)
    assert build_bundle(sid).edges == []
