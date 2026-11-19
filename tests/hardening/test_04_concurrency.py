import random
import threading
import time
from collections import Counter

import pytest
from fastapi.testclient import TestClient

from .conftest import TERMINAL, running_app, topic_texts, upload, wait_job

pytestmark = pytest.mark.hardening
OK_CODES = {200, 202, 204, 400, 404, 409, 422}


def active_jobs_by_session():
    from sqlalchemy import func, select
    from reqcluster.db import session_scope
    from reqcluster.models import JobRow
    with session_scope() as s:
        return dict(s.execute(select(JobRow.session_id, func.count()).where(
            JobRow.status.in_(("queued", "running"))).group_by(JobRow.session_id)).all())


def all_jobs():
    from sqlalchemy import select
    from reqcluster.db import session_scope
    from reqcluster.models import JobRow
    with session_scope() as s:
        return [(j.id, j.session_id, j.kind, j.status, j.message) for j in s.scalars(select(JobRow))]


def worker(app, sids, seed, n_ops, codes, errors):
    c = TestClient(app, raise_server_exceptions=False)
    rnd = random.Random(seed)

    def call(method, url, **kw):
        r = c.request(method, url, **kw)
        codes[(method, url.split("?")[0].rstrip("0123456789").rstrip("/") or url, r.status_code)] += 1
        if r.status_code == 500 and method == "POST" and url.startswith(f"/api/sessions/{sids[-1]}/"):
            return r  # delete-vs-start race, pinned deterministically by test_submit_for_deleted_session
        if r.status_code not in OK_CODES:
            errors.append((method, url, r.status_code, r.text[:200]))
        return r

    for _ in range(n_ops):
        sid = rnd.choice(sids)
        base = f"/api/sessions/{sid}"
        op = rnd.choice(["cluster", "cluster", "recluster", "enrich", "suggest", "deps", "move", "move",
                         "accept", "read", "read", "readall", "delete", "graph"])
        if op == "cluster":
            call("POST", base + "/cluster", json={"min_cluster_size": rnd.choice([2, 3, 5])})
        elif op == "recluster":
            call("POST", base + "/recluster")
        elif op == "enrich":
            call("POST", base + "/enrich", json={})
        elif op == "suggest":
            call("POST", base + "/suggestions", json={"min_silhouette_delta": None})
        elif op == "deps":
            call("POST", base + "/dependencies", json={})
        elif op == "move":
            reqs = call("GET", base + "/requirements?page_size=1000")
            cl = call("GET", base + "/clusters")
            if reqs.status_code == 200 and cl.status_code == 200 and reqs.json():
                ids = [x["cluster_id"] for x in cl.json()]
                call("POST", base + "/feedback", json={
                    "requirement_id": rnd.choice(reqs.json())["id"],
                    "new_cluster_id": rnd.choice(ids + [None]) if ids else None})
        elif op == "accept":
            sg = call("GET", base + "/suggestions?status=pending")
            if sg.status_code == 200 and sg.json():
                call("POST", f"{base}/suggestions/{rnd.choice(sg.json())['id']}/accept", json={})
        elif op == "read":
            call("GET", base + "/requirements?page_size=50&sort=probability&dir=desc")
        elif op == "readall":
            for p in ("", "/clusters", "/metrics", "/uncertainty", "/coherence", "/audit", "/dependencies"):
                call("GET", base + p)
        elif op == "graph":
            call("GET", base + "/graph")
        elif op == "delete" and rnd.random() < 0.08:
            call("DELETE", f"/api/sessions/{sids[-1]}")  # only one sacrificial session


def coherent(c, sid):
    s = c.get(f"/api/sessions/{sid}")
    if s.status_code == 404:
        return
    s = s.json()
    reqs, page = [], 1
    while True:
        g = c.get(f"/api/sessions/{sid}/requirements", params={"page": page, "page_size": 1000})
        reqs += g.json()
        if len(reqs) >= int(g.headers["x-total-count"]) or not g.json():
            break
        page += 1
    assert len(reqs) == s["total_requirements"], (sid, "requirement count")
    assert len({r["id"] for r in reqs}) == len(reqs)
    if s["revision"] == 0 and s["total_clusters"] == 0 and all(r["cluster_id"] is None for r in reqs):
        return  # never clustered
    cl = c.get(f"/api/sessions/{sid}/clusters")
    assert cl.status_code == 200, (sid, cl.text)
    cl = cl.json()
    real = {x["cluster_id"]: x["size"] for x in cl if x["cluster_id"] != -1}
    count = Counter(r["cluster_id"] for r in reqs)
    noise = count.pop(-1, 0)
    assert None not in count, (sid, "requirement without assignment after quiescence")
    assert set(count) <= set(real), (sid, "assignment to unknown cluster", set(count) - set(real))
    for cid, size in real.items():
        assert count.get(cid, 0) == size, (sid, cid, "cluster size != assigned", size, count.get(cid, 0))
    assert sum(real.values()) + noise == s["total_requirements"], (sid, "sizes+noise != total")
    assert s["total_clusters"] == len(real), (sid, "session total_clusters", s["total_clusters"], len(real))
    assert s["noise_count"] == noise, (sid, "session noise_count", s["noise_count"], noise)


@pytest.mark.parametrize("workers", [1, 2])
def test_concurrent_mixed_operations(tmp_path, workers):
    with running_app(tmp_path, workers=workers) as (c, _):
        warm = upload(c, topic_texts(8, tag=" warm"))  # compile numba before timing-sensitive stress
        assert wait_job(c, c.post(f"/api/sessions/{warm}/cluster", json={}).json()["id"], 120)["status"] == "succeeded"
        c.delete(f"/api/sessions/{warm}")
        sids = [upload(c, topic_texts(8, tag=f" s{i}")) for i in range(3)]
        codes, errors, overlaps = Counter(), [], []
        stop = threading.Event()

        def monitor():
            while not stop.is_set():
                bad = {k: v for k, v in active_jobs_by_session().items() if v > 1}
                if bad:
                    overlaps.append(bad)
                time.sleep(0.01)

        mt = threading.Thread(target=monitor, daemon=True)
        mt.start()
        ts = [threading.Thread(target=worker, args=(c.app, sids, 100 * workers + i, 40, codes, errors))
              for i in range(5)]
        [t.start() for t in ts]
        [t.join(240) for t in ts]
        assert not any(t.is_alive() for t in ts), "worker threads hung"
        end = time.time() + 120
        while active_jobs_by_session() and time.time() < end:
            time.sleep(0.1)
        stop.set()
        assert not errors, errors[:5]
        assert not overlaps, f"more than one active job per session: {overlaps[:3]}"
        jobs = all_jobs()
        assert all(j[3] in TERMINAL for j in jobs), [j for j in jobs if j[3] not in TERMINAL]
        assert sum(1 for j in jobs if j[3] == "succeeded") >= 6, Counter(j[3] for j in jobs)
        assert {j[2] for j in jobs} >= {"cluster", "enrich"}, Counter(j[2] for j in jobs)
        for sid in sids:
            coherent(c, sid)
        print(workers, dict(Counter(j[3] for j in jobs)), sum(v for k, v in codes.items() if k[2] == 409), "x409")


@pytest.mark.xfail(strict=True, reason="FLAW: jobs.py:88-92 submit() only translates the uq_jobs_one_active "
                   "IntegrityError; a session deleted between the service's existence check and the job "
                   "insert surfaces as FOREIGN KEY IntegrityError -> 500 (seen as POST /suggestions 500 "
                   "under concurrent DELETE). Fix: map FK violations to NotFound in submit()")
def test_submit_for_deleted_session_is_not_found(app):
    from reqcluster.errors import NotFound
    from reqcluster.jobs import get_runner
    with pytest.raises(NotFound):
        get_runner().submit("cluster", 424242, lambda ctx: None)
