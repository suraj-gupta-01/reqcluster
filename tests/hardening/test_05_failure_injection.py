import sqlite3
import time

import pytest
from sqlalchemy import event
from sqlalchemy.exc import OperationalError

from .conftest import (FakeLLM, TERMINAL, is_envelope, run_cluster, running_app, topic_texts, upload,
                       wait_job)

pytestmark = pytest.mark.hardening


def test_embedding_failure_job_failed_session_usable(app_async):
    c, emb = app_async
    sid = upload(c, topic_texts(5))
    emb.fail = RuntimeError("CUDA out of memory")
    j = run_cluster(c, sid)
    assert j["status"] == "failed" and j["message"]
    assert "Traceback" not in j["message"]
    assert c.get(f"/api/sessions/{sid}").json()["status"] == "error"
    emb.fail = None
    assert run_cluster(c, sid)["status"] == "succeeded"
    assert c.get(f"/api/sessions/{sid}").json()["status"] == "done"
    assert c.get(f"/api/sessions/{sid}/clusters").status_code == 200


def test_restart_sweeps_interrupted_job(tmp_path):
    from reqcluster import db, jobs
    from reqcluster.app import create_app
    from fastapi.testclient import TestClient
    with running_app(tmp_path, workers=1) as (c, emb):
        sid = upload(c, topic_texts(5))
        emb.hold()
        r = c.post(f"/api/sessions/{sid}/cluster", json={})
        assert r.status_code == 202
        jid = r.json()["id"]
        assert emb.entered.wait(20)
        # simulate a crash: drop the runner/engine without finishing the job
        jobs._runner = None
        db.reset()
        with TestClient(create_app(), raise_server_exceptions=False) as c2:
            j = c2.get(f"/api/jobs/{jid}").json()
            assert j["status"] == "failed" and "interrupted by restart" in (j["message"] or ""), j
            assert c2.get(f"/api/sessions/{sid}").json()["status"] == "error"
            emb.release()
            time.sleep(3)  # let the orphaned thread of the "dead process" finish before reuse
            assert run_cluster(c2, sid)["status"] == "succeeded"


def _lock_errors(match, times):
    state = {"n": 0}

    def hook(conn, cursor, statement, parameters, context, executemany):
        if match(statement) and state["n"] < times:
            state["n"] += 1
            raise OperationalError(statement, parameters, sqlite3.OperationalError("database is locked"))
    return hook, state


@pytest.mark.parametrize("times", [1, 10**6])
def test_db_locked_during_result_write(app_async, times):
    from reqcluster import db
    c, _ = app_async
    sid = upload(c, topic_texts(5))
    hook, state = _lock_errors(lambda s: s.lstrip().upper().startswith("INSERT INTO ASSIGNMENTS"), times)
    event.listen(db.get_engine(), "before_cursor_execute", hook)
    try:
        j = run_cluster(c, sid)
        assert j["status"] in TERMINAL
        if times > 1:
            assert j["status"] == "failed" and j["message"]
        assert state["n"] >= 1
    finally:
        event.remove(db.get_engine(), "before_cursor_execute", hook)
    assert run_cluster(c, sid)["status"] == "succeeded"  # session not wedged


def _final_write_lock(c, times):
    from reqcluster import db
    # 16 rows: below the UMAP threshold (20), so a cold process does not spend the wait window
    # compiling numba; this test is about lock recovery, not JIT time.
    sid = upload(c, topic_texts(4))
    hook, _ = _lock_errors(lambda s: s.lstrip().upper().startswith("UPDATE JOBS") and "finished_at" in s, times)
    event.listen(db.get_engine(), "before_cursor_execute", hook)
    jid = c.post(f"/api/sessions/{sid}/cluster", json={}).json()["id"]
    try:
        end = time.time() + 15
        while time.time() < end and c.get(f"/api/jobs/{jid}").json()["status"] not in TERMINAL:
            time.sleep(0.2)
    finally:
        event.remove(db.get_engine(), "before_cursor_execute", hook)
    time.sleep(1)  # lock gone: a correct implementation has settled by now
    return sid, c.get(f"/api/jobs/{jid}").json()["status"]


def test_db_locked_briefly_on_final_status_write(app_async):
    _, status = _final_write_lock(app_async[0], 2)
    assert status in TERMINAL, f"job stuck in {status!r}"


def test_db_locked_long_on_final_status_write(app_async):
    sid, status = _final_write_lock(app_async[0], 10**6)
    assert status in TERMINAL, f"job stuck in {status!r}"


@pytest.mark.parametrize("mode", ["401", "500", "garbage", "html", "down"])
def test_llm_failure_enrichment_fallback(tmp_path, mode):
    srv = FakeLLM(mode if mode != "down" else "ok")
    url = srv.url if mode != "down" else "http://127.0.0.1:9/v1"
    if mode == "down":
        srv.close()
    env = {"REQCLUSTER_LLM_PROVIDER": "openai", "REQCLUSTER_LLM_BASE_URL": url, "REQCLUSTER_LLM_API_KEY": "k",
           "REQCLUSTER_LLM_MODEL": "m", "REQCLUSTER_LLM_MAX_RETRIES": "0",
           "REQCLUSTER_LLM_TIMEOUT_SECONDS": "3"}
    try:
        with running_app(tmp_path, workers=1, env=env) as (c, _):
            sid = upload(c, topic_texts(3))
            r = c.post(f"/api/sessions/{sid}/enrich", json={})
            assert r.status_code == 202
            j = wait_job(c, r.json()["id"], 90)
            assert j["status"] == "succeeded", j
            assert j["result"]["total"] == 12
            assert j["result"]["fallback"] == 12, j["result"]
            summ = c.get(f"/api/sessions/{sid}/enrichment/summary")
            assert summ.status_code == 200
            assert c.get(f"/api/sessions/{sid}/enrichment").status_code == 200
            r = c.post(f"/api/sessions/{sid}/cluster", json={"embedding_mode": "enriched"})
            assert r.status_code == 409 and is_envelope(r), (r.status_code, r.text)
            assert run_cluster(c, sid)["status"] == "succeeded"  # base mode still works
    finally:
        if mode != "down":
            srv.close()
