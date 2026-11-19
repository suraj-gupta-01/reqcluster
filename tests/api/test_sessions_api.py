import pytest

TOPICS = ["brake pedal caliper hydraulic", "battery cell charger voltage",
          "display screen pixel backlight"]


def csv_bytes(n=30):
    rows = ["id,text"]
    for i in range(n):
        rows.append(f"R{i},The {TOPICS[i % 3]} subsystem shall report status code {i} quickly")
    return "\n".join(rows).encode()


def upload(client, n=30):
    r = client.post("/api/sessions", files={"file": ("r.csv", csv_bytes(n))})
    assert r.status_code == 200, r.text
    return r.json()["session"]["id"]


def test_flow(client):
    sid = upload(client)
    r = client.post(f"/api/sessions/{sid}/cluster")
    assert r.status_code == 202 and r.json()["status"] == "succeeded"
    assert client.get(f"/api/jobs/{r.json()['id']}").json()["status"] == "succeeded"
    s = client.get(f"/api/sessions/{sid}").json()
    assert s["status"] == "done" and s["revision"] == 1
    assert client.get(f"/api/sessions/{sid}/clusters").status_code == 200
    r = client.get(f"/api/sessions/{sid}/requirements", params={"page_size": 10})
    assert len(r.json()) == 10 and r.headers["x-total-count"] == "30"
    g = client.get(f"/api/sessions/{sid}/graph").json()
    n = g["nodes"]
    assert len(n["id"]) == 30 and all(len(v) == 30 for v in n.values())
    assert set(g["edges"]) == {"source", "target", "weight"} and "text" not in n
    assert g["edit_seq"] == 0 and g["revision"] == 1
    assert client.get(f"/api/sessions/{sid}/graph").json() == g  # cached path
    assert "metrics" in client.get(f"/api/sessions/{sid}/metrics").json()
    assert client.delete(f"/api/sessions/{sid}").status_code == 204


def test_errors(client):
    r = client.get("/api/sessions/99")
    assert r.status_code == 404 and r.json()["error"] == "not_found"
    sid = upload(client)
    r = client.get(f"/api/sessions/{sid}/clusters")
    assert r.status_code == 409 and r.json()["error"] == "conflict"
    r = client.get("/api/nope")
    assert r.status_code == 404 and r.json()["error"] == "http_error"
    r = client.get(f"/api/sessions/{sid}/requirements", params={"sort": "bogus"})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_too_large(client, monkeypatch):
    from reqcluster.app import create_app
    from reqcluster.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("REQCLUSTER_MAX_UPLOAD_MB", "1")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        r = c.post("/api/sessions", files={"file": ("r.csv", b"a" * (1024 * 1024 + 10))})
    assert r.status_code == 413 and r.json()["error"] == "too_large"


@pytest.mark.parametrize("key,expected", [(None, 401), ("k", 200)])
def test_api_key(client, monkeypatch, key, expected):
    from reqcluster.app import create_app
    from reqcluster.config import get_settings
    from fastapi.testclient import TestClient

    monkeypatch.setenv("REQCLUSTER_API_KEY", "k")
    get_settings.cache_clear()
    with TestClient(create_app()) as c:
        r = c.get("/api/sessions", headers={"X-API-Key": key} if key else {})
        assert r.status_code == expected
        if expected == 401:
            assert r.json()["error"] == "unauthorized"
        assert c.get("/health").status_code == 200


def test_health_mock(client):
    assert client.get("/health").json() == {"status": "ok", "embedder": "mock", "is_mock": True} \
        or client.get("/health").json()["is_mock"] is True
