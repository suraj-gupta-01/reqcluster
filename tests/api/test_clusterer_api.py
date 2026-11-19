from .test_sessions_api import upload


def test_ward_accepted_unknown_rejected(client):
    sid = upload(client)
    r = client.post(f"/api/sessions/{sid}/cluster", json={"clusterer": "ward"})
    assert r.status_code == 202 and r.json()["status"] == "succeeded"
    r = client.post(f"/api/sessions/{sid}/cluster", json={"clusterer": "kmeans"})
    assert r.status_code == 422
