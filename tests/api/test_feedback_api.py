from .test_sessions_api import upload


def setup(client):
    sid = upload(client)
    assert client.post(f"/api/sessions/{sid}/cluster").status_code == 202
    reqs = client.get(f"/api/sessions/{sid}/requirements", params={"page_size": 100}).json()
    rid = reqs[0]["id"]
    dst = next(r["cluster_id"] for r in reqs if r["cluster_id"] not in (-1, reqs[0]["cluster_id"]))
    return sid, rid, dst


def test_flow(client):
    sid, rid, dst = setup(client)
    u = f"/api/sessions/{sid}"
    r = client.post(f"{u}/feedback", json={"requirement_id": rid, "new_cluster_id": dst})
    assert r.status_code == 200 and r.json()["status"] == "pending"
    fid = r.json()["feedback_id"]
    assert len(client.get(f"{u}/feedback", params={"status": "pending"}).json()) == 1
    assert client.get(f"{u}/constraints").json()["constraint_pairs"]
    assert client.post(f"{u}/feedback/{fid}/reject").json()["status"] == "rejected"
    assert client.get(f"{u}/constraints").json()["constraint_pairs"] == []
    assert client.post(f"{u}/feedback/9999/approve").status_code == 404
    assert client.post(f"{u}/feedback", json={"requirement_id": rid, "new_cluster_id": 999}
                       ).status_code == 404
    r = client.post(f"{u}/feedback", json={"requirement_id": rid, "new_cluster_id": None})
    assert r.json()["new_cluster_id"] == -1
    assert client.get(f"{u}/feedback/export", params={"format": "json"}).status_code == 200
    assert "status" in client.get(f"{u}/feedback/export").text
    j = client.post(f"{u}/recluster")
    assert j.status_code == 202 and j.json()["status"] == "succeeded"
    assert client.get(f"{u}").json()["revision"] == 2


def test_active_learning(client):
    sid, _, _ = setup(client)
    u = f"/api/sessions/{sid}"
    assert len(client.get(f"{u}/uncertainty", params={"top_k": 3}).json()) == 3
    assert client.get(f"{u}/quality").json()[0]["revision"] == 1
    assert client.get(f"{u}/uncertainty", params={"top_k": 0}).status_code == 422
    assert client.get("/api/sessions/999/quality").status_code == 404
