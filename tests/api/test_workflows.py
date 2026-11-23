"""Workflow tests: chain several steps, then check every read endpoint tells the same story.

Earlier tests asserted one feature at a time ("200, no data loss"). The bugs they missed sat
between features and returned 200 with wrong content, so each step here is followed by
`assert_consistent`, which cross-checks requirements, clusters, cluster detail, graph,
metrics, session summary and the CSV export against each other.
"""
import csv
import io
import json
from collections import Counter

from .test_sessions_api import TOPICS, upload

LOOSE = {"sim_threshold": 0.0, "spread_threshold": 0.0, "min_silhouette_delta": None}


def reqs(client, sid, **params):
    return client.get(f"/api/sessions/{sid}/requirements", params={"page_size": 1000, **params}).json()


def placement(client, sid):
    """{requirement pk: cluster_id} as the requirements endpoint reports it."""
    return {r["id"]: r["cluster_id"] for r in reqs(client, sid)}


def assert_consistent(client, sid):
    u = f"/api/sessions/{sid}"
    items = reqs(client, sid)
    by_pk = {r["id"]: r for r in items}
    cid_of = {r["id"]: r["cluster_id"] for r in items}
    sizes = Counter(c for c in cid_of.values() if c != -1)
    noise = {pk for pk, c in cid_of.items() if c == -1}

    clusters = client.get(f"{u}/clusters").json()
    assert {c["cluster_id"]: c["size"] for c in clusters} == dict(sizes)
    for c in clusters:
        detail = client.get(f"{u}/clusters/{c['cluster_id']}").json()
        assert {r["id"] for r in detail["requirements"]} == {pk for pk, x in cid_of.items()
                                                            if x == c["cluster_id"]}
        assert detail["size"] == c["size"] and detail["label"] == c["label"]
        assert {r["id"] for r in reqs(client, sid, cluster_id=c["cluster_id"])} == \
            {r["id"] for r in detail["requirements"]}
    assert {r["id"] for r in reqs(client, sid, noise=True)} == noise
    for r in items:
        assert r["is_noise"] == (r["cluster_id"] == -1)
        if r["cluster_id"] == -1:
            assert r["probability"] == 0.0

    g = client.get(f"{u}/graph").json()["nodes"]
    assert dict(zip(g["id"], g["cluster_id"], strict=True)) == cid_of

    m = client.get(f"{u}/metrics").json()
    assert m["n_clusters"] == len(clusters) == m["metrics"]["n_clusters"]
    assert m["noise_count"] == len(noise) == m["metrics"]["noise_count"]
    s = client.get(u).json()
    assert s["total_clusters"] == len(clusters) and s["noise_count"] == len(noise)
    assert s["total_requirements"] == len(items)

    labels = {c["cluster_id"]: c["label"] for c in clusters}
    rows = list(csv.DictReader(io.StringIO(client.get(f"{u}/export/csv").content.decode("utf-8-sig"))))
    assert len(rows) == len(items)
    for row in rows:
        r = next(x for x in items if x["req_id"] == row["req_id"])
        assert int(row["cluster_id"]) == r["cluster_id"]
        assert row["is_noise"] == str(r["cluster_id"] == -1)
        assert row["cluster_label"] == ("Noise" if r["cluster_id"] == -1 else labels[r["cluster_id"]])
    return by_pk


def clustered(client, n=30, **params):
    sid = upload(client, n)
    assert client.post(f"/api/sessions/{sid}/cluster", json=params).json()["status"] == "succeeded"
    assert_consistent(client, sid)
    return sid


def test_move_reject_and_noise_round_trip(client):
    sid = clustered(client)
    u = f"/api/sessions/{sid}"
    before = placement(client, sid)
    rid = next(pk for pk, c in before.items() if c != -1)
    dst = next(c for c in set(before.values()) if c not in (-1, before[rid]))

    fid = client.post(f"{u}/feedback", json={"requirement_id": rid, "new_cluster_id": dst}).json()["feedback_id"]
    after = assert_consistent(client, sid)
    assert after[rid]["cluster_id"] == dst and after[rid]["probability"] == 1.0
    assert {pk: c for pk, c in placement(client, sid).items() if pk != rid} == \
        {pk: c for pk, c in before.items() if pk != rid}

    client.post(f"{u}/feedback/{fid}/reject")
    assert placement(client, sid) == before
    assert_consistent(client, sid)

    client.post(f"{u}/feedback", json={"requirement_id": rid, "new_cluster_id": None})
    after = assert_consistent(client, sid)
    assert after[rid]["cluster_id"] == -1 and after[rid]["probability"] == 0.0
    audit = [a["action"] for a in client.get(f"{u}/audit").json()]
    assert Counter(audit) == {"feedback_move": 2, "feedback_reject": 1}


def test_emptying_a_cluster_removes_it_everywhere(client):
    sid = clustered(client)
    u = f"/api/sessions/{sid}"
    before = placement(client, sid)
    victim = min(Counter(c for c in before.values() if c != -1).items(), key=lambda kv: kv[1])[0]
    dst = next(c for c in set(before.values()) if c not in (-1, victim))
    for pk in [pk for pk, c in before.items() if c == victim]:
        assert client.post(f"{u}/feedback", json={"requirement_id": pk, "new_cluster_id": dst}).status_code == 200
    assert_consistent(client, sid)
    assert victim not in {c["cluster_id"] for c in client.get(f"{u}/clusters").json()}
    assert client.get(f"{u}/clusters/{victim}").status_code == 404
    assert client.post(f"{u}/feedback", json={"requirement_id": next(iter(before)),
                                              "new_cluster_id": victim}).status_code == 404


def test_feedback_survives_recluster_and_stays_reviewable(client):
    sid = clustered(client)
    u = f"/api/sessions/{sid}"
    before = placement(client, sid)
    rid = next(pk for pk, c in before.items() if c != -1)
    dst = next(c for c in set(before.values()) if c not in (-1, before[rid]))
    mates = {pk for pk, c in before.items() if c == dst}
    client.post(f"{u}/feedback", json={"requirement_id": rid, "new_cluster_id": dst})
    assert client.post(f"{u}/recluster").json()["status"] == "succeeded"

    s = client.get(u).json()
    assert s["revision"] == 2
    after = assert_consistent(client, sid)
    # the correction is honoured: the moved item still sits with its new cluster-mates
    assert {after[pk]["cluster_id"] for pk in mates | {rid}} == {after[rid]["cluster_id"]}
    fb = client.get(f"{u}/feedback").json()
    assert [(f["requirement_id"], f["status"]) for f in fb] == [(rid, "pending")]
    assert client.get(f"{u}/constraints").json()["constraint_pairs"]
    r = client.post(f"{u}/feedback/{fb[0]['id']}/approve")
    assert r.status_code == 200 and r.json()["status"] == "approved"


def test_merge_then_split_keep_views_in_sync(client):
    sid = clustered(client, 60, min_cluster_size=5)
    u = f"/api/sessions/{sid}"
    client.post(f"{u}/suggestions", json=LOOSE)
    merge = next(i for i in client.get(f"{u}/suggestions", params={"status": "pending"}).json()
                 if i["kind"] == "merge")
    a, b = merge["payload"]["cluster_a"], merge["payload"]["cluster_b"]
    before = placement(client, sid)
    client.post(f"{u}/suggestions/{merge['id']}/accept")
    after = assert_consistent(client, sid)
    for pk, c in before.items():
        assert after[pk]["cluster_id"] == (a if c == b else c)
    assert all(i["stale"] for i in client.get(f"{u}/suggestions", params={"status": "pending"}).json())

    client.post(f"{u}/suggestions", json=LOOSE)
    split = next((i for i in client.get(f"{u}/suggestions", params={"status": "pending"}).json()
                  if i["kind"] == "split"), None)
    assert split is not None
    before = placement(client, sid)
    client.post(f"{u}/suggestions/{split['id']}/accept")
    after = assert_consistent(client, sid)
    moved = {int(k) for k, v in split["payload"]["assignment"].items() if v == 1}
    assert len({after[pk]["cluster_id"] for pk in moved}) == 1
    assert {after[pk]["cluster_id"] for pk in moved}.isdisjoint(set(before.values()))
    assert all(after[pk]["cluster_id"] == c for pk, c in before.items() if pk not in moved)


def test_dependency_edges_follow_edits_into_exports(client):
    rows = ["id,text"] + [f"R{i},The {TOPICS[i % 3]} subsystem shall report code {i}"
                          + (f" as defined in R{i - 1}" if i else "") for i in range(30)]
    sid = client.post("/api/sessions", files={"file": ("r.csv", "\n".join(rows).encode())}
                      ).json()["session"]["id"]
    u = f"/api/sessions/{sid}"
    client.post(f"{u}/cluster")
    client.post(f"{u}/dependencies", json={"sim_threshold": 0.0})
    tree = client.get(f"{u}/dependencies").json()
    ids = {r["id"]: r["req_id"] for r in reqs(client, sid)}
    want = {(ids[e["source"]], ids[e["target"]]) for e in tree["edges"]}
    jama = json.loads(client.get(f"{u}/export/jama").content)
    assert want and {(r["fromItem"], r["toItem"]) for r in jama["relationships"]} == want

    before = placement(client, sid)
    rid = next(pk for pk, c in before.items() if c != -1)
    dst = next(c for c in set(before.values()) if c not in (-1, before[rid]))
    client.post(f"{u}/feedback", json={"requirement_id": rid, "new_cluster_id": dst})
    assert client.get(f"{u}/dependencies").json()["stale"] is True
    assert json.loads(client.get(f"{u}/export/jama").content)["relationships"] == []

    client.post(f"{u}/dependencies", json={"sim_threshold": 0.0})
    fresh = client.get(f"{u}/dependencies").json()
    assert fresh["stale"] is False
    assert len(json.loads(client.get(f"{u}/export/jama").content)["relationships"]) == len(fresh["edges"])


def test_unclustered_and_deleted_sessions(client):
    sid = upload(client, 10)
    u = f"/api/sessions/{sid}"
    items = reqs(client, sid)
    assert all(r["cluster_id"] is None and r["is_noise"] is None for r in items)
    rows = list(csv.DictReader(io.StringIO(client.get(f"{u}/export/csv").content.decode("utf-8-sig"))))
    assert [r["cluster_label"] for r in rows] == ["Unclustered"] * 10
    for path in ("clusters", "metrics", "graph", "uncertainty", "constraints"):
        assert client.get(f"{u}/{path}").status_code == 409, path

    assert client.delete(u).status_code == 204
    assert sid not in {s["id"] for s in client.get("/api/sessions").json()}
    for path in ("", "/requirements", "/export/csv"):
        assert client.get(f"{u}{path}").status_code == 404, path


def _collapse(client, sid, target):
    """Move every requirement to `target` (a cluster id, or None for noise)."""
    u = f"/api/sessions/{sid}"
    for pk, c in placement(client, sid).items():
        want = -1 if target is None else target
        if c != want:
            r = client.post(f"{u}/feedback", json={"requirement_id": pk, "new_cluster_id": target})
            assert r.status_code == 200, r.text


def test_zero_and_one_cluster_states(client):
    """Every requirement in noise (0 clusters), then all in one cluster: reports, analyses and a
    recluster must still work and agree. Collapsing into one cluster chains many moves out of the
    same source, which used to leave contradictory must/cannot constraints and fail the recluster."""
    for target in (None, "first"):
        sid = clustered(client)
        u = f"/api/sessions/{sid}"
        first = min(c for c in placement(client, sid).values() if c != -1)
        _collapse(client, sid, None if target is None else first)
        assert_consistent(client, sid)
        for fmt in ("csv", "reqif", "sysml", "jama", "pdf"):
            assert client.get(f"{u}/export/{fmt}").status_code == 200, fmt
        for path in ("uncertainty", "quality", "coherence", "audit", "feedback/export"):
            assert client.get(f"{u}/{path}").status_code == 200, path
        assert client.get(f"{u}/constraints").json()["conflicts"] == []
        for path, body in (("suggestions", LOOSE), ("dependencies", {}), ("recluster", None)):
            j = client.post(f"{u}/{path}", json=body).json()
            assert j["status"] == "succeeded", (path, j.get("error"))
        after = assert_consistent(client, sid)
        want = {-1} if target is None else {first}
        assert {r["cluster_id"] for r in after.values()} == want
