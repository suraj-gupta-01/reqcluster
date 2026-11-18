"""7. Progress / jobs."""
import pytest

from .conftest import topic_texts

pytestmark = pytest.mark.acceptance


def test_progress_after_restart_never_returns_previous_done(clustered, api, embedder):
    sid = clustered
    assert api.progress(sid)["state"] == "done"  # primes any progress cache with "done"
    embedder.hold()
    run = api.cluster_start(sid, min_cluster_size=3)
    try:
        assert embedder.entered.wait(30), "pipeline never reached the embedder"
        p = api.progress(sid)
        assert p["state"] in ("running", "queued", "idle") and p["state"] != "done", p
        st, s = api.session(sid)
        assert s["status"] != "done", s
    finally:
        embedder.release()
    assert api.wait(run)["state"] == "done"
    assert api.progress(sid)["state"] == "done"


def test_failure_surfaces_as_error_with_message(api, embedder):
    sid = api.upload_texts(topic_texts(3))
    embedder.fail = RuntimeError("embedding backend exploded")
    res = api.cluster(sid)
    assert res["state"] == "error", res
    p = api.progress(sid)
    assert p["state"] == "error"
    assert isinstance(p["message"], str) and p["message"].strip()
    assert api.session(sid)[1]["status"] == "error"


def test_session_recovers_after_failed_run(api, embedder):
    sid = api.upload_texts(topic_texts(3))
    embedder.fail = RuntimeError("boom")
    assert api.cluster(sid)["state"] == "error"
    embedder.fail = None
    res = api.cluster(sid)
    assert res["state"] == "done", res


def test_cluster_start_returns_pollable_job(api):
    sid = api.upload_texts(topic_texts(3))
    run = api.cluster_start(sid)
    assert run.start_status in (200, 202)
    job_id = run.job_id
    assert job_id is not None, "start response must carry a job id"
    st, job = api.job(job_id)
    assert st == 200 and job
    api.wait(run)
