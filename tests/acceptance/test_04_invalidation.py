"""4. Re-cluster invalidates derived artefacts."""
import pytest

from .conftest import DEP_TEXTS

pytestmark = pytest.mark.acceptance


def _gen_pending(api, sid):
    st, b = api.suggest(sid, sim_threshold=0.0, spread_threshold=0.0)
    assert st == 200, b
    st, items = api.suggestions(sid)
    assert st == 200 and items, "precondition: at least one suggestion"
    return [s["id"] for s in items]


def test_suggestions_not_served_as_current_after_recluster(clustered, api):
    sid = clustered
    old = set(_gen_pending(api, sid))
    assert api.cluster(sid, min_cluster_size=3)["state"] == "done"
    st, items = api.suggestions(sid)
    assert st in (404, 409) or (st == 200 and all(
        s["status"] != "pending" or s["id"] not in old for s in items)), (st, items)


def test_apply_stale_suggestion_409(clustered, api):
    sid = clustered
    sug = _gen_pending(api, sid)[0]
    assert api.cluster(sid, min_cluster_size=3)["state"] == "done"
    st, b = api.apply_suggestion(sid, sug, "accept")
    assert st == 409, (st, b)


def test_feedback_constraints_not_served_after_recluster(clustered, api):
    sid = clustered
    rid = api.all_requirements(sid)[0]["id"]
    st, b = api.move(sid, rid, -1)
    assert st == 200, b
    st, c = api.get("constraints", sid)
    assert st == 200 and c["constraint_pairs"], "precondition: constraints exist"
    assert api.cluster(sid)["state"] == "done"
    st, c = api.get("constraints", sid)
    assert st in (404, 409) or c["constraint_pairs"] == [], (st, c)
    st, q = api.get("feedback_queue", sid)
    assert st in (404, 409) or all(f["status"] != "pending" for f in q), (st, q)


def test_dependency_tree_and_explanations_not_served_after_recluster(dep_session, api):
    sid = dep_session
    st, deps = api.dependencies(sid)
    e = deps["edges"][0]
    st, ex = api.explain(sid, e["source"], e["target"], provider_name="mock")
    assert st == 200, ex
    assert api.cluster(sid, min_cluster_size=2)["state"] == "done"
    st, deps2 = api.dependencies(sid)
    assert st in (404, 409) or deps2["edges"] != [], (st, deps2)
    st, ex2 = api.explain(sid, e["source"], e["target"], provider_name="mock")
    if st == 200:
        # regenerated tree is acceptable, but a cached explanation from the old run is not
        assert ex2["cached"] is False, ex2
    else:
        assert st in (404, 409)
