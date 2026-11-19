from .test_sessions_api import upload

LOOSE = {"sim_threshold": 0.0, "spread_threshold": 0.0, "min_silhouette_delta": None}


def prep(client):
    sid = upload(client, 60)
    client.post(f"/api/sessions/{sid}/cluster", json={"min_cluster_size": 5})
    return sid


def test_flow(client):
    sid = prep(client)
    r = client.post(f"/api/sessions/{sid}/suggestions", json=LOOSE)
    assert r.status_code == 202 and r.json()["status"] == "succeeded"
    assert r.json()["kind"] == "suggest" and r.json()["result"]["n_merges"] >= 1
    items = client.get(f"/api/sessions/{sid}/suggestions", params={"status": "pending"}).json()
    assert items and set(items[0]) >= {"id", "kind", "status", "payload", "rationale",
                                       "provider", "created_at", "stale"}
    n = client.get(f"/api/sessions/{sid}").json()["total_clusters"]
    merge = next(i for i in items if i["kind"] == "merge")
    r = client.post(f"/api/sessions/{sid}/suggestions/{merge['id']}/accept", json={"actor": "bob"})
    assert r.status_code == 200 and r.json()["status"] == "accepted"
    assert client.get(f"/api/sessions/{sid}").json()["total_clusters"] == n - 1
    audit = client.get(f"/api/sessions/{sid}/audit").json()
    assert audit[0]["actor"] == "bob"
    other = next(i for i in items if i["id"] != merge["id"])
    r = client.post(f"/api/sessions/{sid}/suggestions/{other['id']}/reject")
    assert r.status_code == 409 and r.json()["error"] == "stale"


def test_errors(client):
    sid = upload(client, 30)
    assert client.post(f"/api/sessions/{sid}/suggestions").status_code == 409
    sid = prep(client)
    for body in ({"provider": "nope"}, {"bogus": 1}, {"sim_threshold": 2}):
        assert client.post(f"/api/sessions/{sid}/suggestions", json=body).status_code == 422
    assert client.post(f"/api/sessions/{sid}/suggestions/9999/accept").status_code == 404
    assert client.post("/api/sessions/999/suggestions").status_code == 404
