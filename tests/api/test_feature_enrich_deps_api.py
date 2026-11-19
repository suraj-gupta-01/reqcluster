from .test_sessions_api import TOPICS, upload


def prep(client, n=30):
    sid = upload(client, n)
    client.post(f"/api/sessions/{sid}/cluster")
    return sid


def test_enrich(client):
    sid = upload(client, 5)
    r = client.post(f"/api/sessions/{sid}/enrich", json={})
    assert r.status_code == 202 and r.json()["kind"] == "enrich"
    assert r.json()["result"]["total"] == 5
    out = client.get(f"/api/sessions/{sid}/enrichment").json()
    assert len(out) == 5 and out[0]["status"] == "succeeded"
    assert client.post(f"/api/sessions/{sid}/enrich", json={"provider": "x"}).status_code == 422
    assert client.post("/api/sessions/999/enrich", json={}).status_code == 404
    assert client.get("/api/sessions/999/enrichment").status_code == 404


def test_dependencies(client):
    rows = ["id,text"] + [f"R{i},The {TOPICS[i % 3]} subsystem shall report code {i}"
                          + (f" as defined in R{i - 1}" if i else "") for i in range(30)]
    sid = client.post("/api/sessions", files={"file": ("r.csv", "\n".join(rows).encode())}
                      ).json()["session"]["id"]
    client.post(f"/api/sessions/{sid}/cluster")
    assert client.get(f"/api/sessions/{sid}/dependencies").status_code == 409
    r = client.post(f"/api/sessions/{sid}/dependencies", json={"sim_threshold": 0.0})
    assert r.status_code == 202 and r.json()["kind"] == "dependencies"
    body = client.get(f"/api/sessions/{sid}/dependencies").json()
    assert set(body) == {"run_id", "nodes", "edges", "stats", "rationale", "stale", "warnings"} and body["edges"]
    e = body["edges"][0]
    url = f"/api/sessions/{sid}/dependencies/explain"
    first = client.post(url, json={"source": e["source"], "target": e["target"]})
    assert first.status_code == 200 and first.json()["cached"] is False
    assert client.post(url, json={"source": e["source"], "target": e["target"]}).json()["cached"]
    assert client.post(url, json={"source": 1, "target": 2, "bogus": 1}).status_code == 422
    assert client.post(url, json={"source": -1, "target": -2}).status_code == 404
    assert client.post(f"/api/sessions/{sid}/dependencies",
                       json={"provider": "x"}).status_code == 422
    assert client.post("/api/sessions/999/dependencies", json={}).status_code == 404


def test_export(client):
    sid = prep(client, 10)
    for fmt, ctype in (("csv", "text/csv"), ("reqif", "application/xml")):
        r = client.get(f"/api/sessions/{sid}/export/{fmt}")
        assert r.status_code == 200 and r.headers["content-type"].startswith(ctype)
        assert "attachment; filename=" in r.headers["content-disposition"]
    assert client.get(f"/api/sessions/{sid}/export/nope").status_code == 400
    assert client.get("/api/sessions/999/export/csv").status_code == 404
