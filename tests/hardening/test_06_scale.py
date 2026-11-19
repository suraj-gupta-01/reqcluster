"""35k-row scale run. Run once: uv run pytest -m "hardening and slow" tests/hardening/test_06_scale.py -s"""
import json
import pathlib
import time

import pytest

from .conftest import running_app, wait_job

pytestmark = [pytest.mark.hardening, pytest.mark.slow]
CSV = pathlib.Path(__file__).resolve().parents[2] / "data" / "uav_fms_35k_requirements.csv"


def rss_mb():
    try:
        import resource
    except ImportError:  # Windows
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KB on Linux


@pytest.fixture(scope="module")
def scale(tmp_path_factory):
    if not CSV.exists():
        pytest.skip("data/uav_fms_35k_requirements.csv missing")
    nums = {}
    with running_app(tmp_path_factory.mktemp("scale"), workers=1, fake_embedder=False) as (c, _):
        data = CSV.read_bytes()
        t = time.perf_counter()
        r = c.post("/api/sessions", files={"file": (CSV.name, data)})
        nums["upload_s"] = round(time.perf_counter() - t, 2)
        assert r.status_code == 200, r.text[:300]
        sid = r.json()["session"]["id"]
        nums["rows"] = r.json()["session"]["total_requirements"]
        t = time.perf_counter()
        j = c.post(f"/api/sessions/{sid}/cluster", json={}).json()
        j = wait_job(c, j["id"], 900)
        nums["cluster_job_s"] = round(time.perf_counter() - t, 2)
        assert j["status"] == "succeeded", j
        nums["peak_rss_mb"] = rss_mb()
        reads = {"requirements_1000": f"/api/sessions/{sid}/requirements?page_size=1000",
                 "graph": f"/api/sessions/{sid}/graph", "clusters": f"/api/sessions/{sid}/clusters",
                 "metrics": f"/api/sessions/{sid}/metrics", "export_csv": f"/api/sessions/{sid}/export/csv",
                 "session": f"/api/sessions/{sid}", "requirements_search":
                     f"/api/sessions/{sid}/requirements?search=geofence&sort=text&page=20"}
        for name, url in reads.items():
            t = time.perf_counter()
            g = c.get(url)
            dt = time.perf_counter() - t
            assert g.status_code == 200, (name, g.status_code)
            nums[name] = {"s": round(dt, 3), "bytes": len(g.content)}
        print("SCALE", json.dumps(nums))
        yield nums


def test_upload_under_30s(scale):
    assert scale["upload_s"] < 30, scale


def test_reads_under_2s(scale):
    slow = {k: v["s"] for k, v in scale.items() if isinstance(v, dict) and k != "graph" and v["s"] >= 2}
    assert not slow, slow


@pytest.mark.xfail(strict=True, reason="FLAW: GET /graph at 35k rows takes ~2.5 s and returns 9.4 MB "
                   "(services/sessions.py graph(): per-node ORM tuples + full text[:200] + every edge "
                   "pydantic-validated). Fix: serialise from raw rows / drop text from nodes / cache per revision")
def test_graph_under_2s(scale):
    assert scale["graph"]["s"] < 2, scale["graph"]
