"""5. Noise semantics on manual moves."""
import pytest

from .helpers import assert_coherent, NOISE

pytestmark = pytest.mark.acceptance


def _req(api, sid, rid):
    return next(r for r in api.all_requirements(sid) if r["id"] == rid)


def test_moving_noise_requirement_into_cluster_clears_is_noise(clustered, api):
    sid = clustered
    reqs = api.all_requirements(sid)
    target = reqs[0]["cluster_id"]
    rid = reqs[-1]["id"]
    assert reqs[-1]["cluster_id"] != target
    assert api.move(sid, rid, -1)[0] == 200
    assert _req(api, sid, rid)["is_noise"] is True
    st, b = api.move(sid, rid, target)
    assert st == 200, b
    r = _req(api, sid, rid)
    assert r["is_noise"] is False and r["cluster_id"] == target
    assert_coherent(api, sid)


def test_moving_to_noise_sets_is_noise_and_counts(clustered, api):
    sid = clustered
    rid = api.all_requirements(sid)[0]["id"]
    assert api.move(sid, rid, -1)[0] == 200
    r = _req(api, sid, rid)
    assert r["is_noise"] is True and r["cluster_id"] in NOISE
    assert_coherent(api, sid)


def test_null_and_minus_one_are_identical(clustered, api):
    sid = clustered
    reqs = api.all_requirements(sid)
    a, b = reqs[0], reqs[1]
    s1, b1 = api.move(sid, a["id"], -1)
    s2, b2 = api.move(sid, b["id"], None)
    assert s1 == s2 == 200, (b1, b2)
    ra, rb = _req(api, sid, a["id"]), _req(api, sid, b["id"])
    assert (ra["is_noise"], ra["cluster_id"]) == (rb["is_noise"], rb["cluster_id"])
    assert b1["new_cluster_id"] == b2["new_cluster_id"]
    assert_coherent(api, sid)


def test_noise_to_noise_null_and_minus_one_behave_same(clustered, api):
    sid = clustered
    rid = api.all_requirements(sid)[0]["id"]
    assert api.move(sid, rid, -1)[0] == 200
    s1, _ = api.move(sid, rid, -1)
    s2, _ = api.move(sid, rid, None)
    assert s1 == s2 and s1 < 500
    assert_coherent(api, sid)


def test_move_updates_cluster_sizes(clustered, api):
    sid = clustered
    reqs = api.all_requirements(sid)
    src, dst = reqs[0]["cluster_id"], next(r["cluster_id"] for r in reqs if r["cluster_id"] != reqs[0]["cluster_id"])
    assert api.move(sid, reqs[0]["id"], dst)[0] == 200
    assert_coherent(api, sid)
    sizes = {c["cluster_id"]: c["size"] for c in api.clusters(sid)[1]}
    assert sizes[src] == 9 and sizes[dst] == 11
