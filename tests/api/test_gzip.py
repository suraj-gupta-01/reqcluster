def test_large_json_is_gzipped(client):
    r = client.get("/openapi.json", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200
    assert r.headers["content-encoding"] == "gzip"
