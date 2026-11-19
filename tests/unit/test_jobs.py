import time

import pytest
from pydantic import BaseModel

from reqcluster import db, jobs
from reqcluster.config import get_settings
from reqcluster.errors import AppError, Conflict, Invalid
from reqcluster.models import SessionRow


@pytest.fixture(autouse=True)
def _db():
    db.init_db()
    with db.session_scope() as s:
        s.add_all([SessionRow(id=1, name="a"), SessionRow(id=2, name="b")])


class Out(BaseModel):
    n: int


def _wait(r, jid):
    for _ in range(300):
        j = r.get(jid)
        if j.status in ("succeeded", "failed", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError("timeout")


def test_inline_success():
    j = jobs.get_runner().submit("cluster", 1, lambda ctx: Out(n=3))
    assert j.status == "succeeded" and j.result == {"n": 3} and j.progress == 1.0
    assert j.finished_at is not None


def test_threaded(monkeypatch):
    monkeypatch.setenv("REQCLUSTER_JOB_WORKERS", "1")
    get_settings.cache_clear()
    r = jobs.get_runner()
    j = r.submit("cluster", 1, lambda ctx: {"ok": 1})
    assert _wait(r, j.id).result == {"ok": 1}


def test_progress_throttle():
    def fn(ctx):
        for i in range(5):
            ctx.progress(0.001 * i, "s")
        ctx.progress(0.5, "s")
        ctx.progress(0.5, "t", "m")

    j = jobs.get_runner().submit("cluster", 1, fn)
    assert j.step == "t" and j.status == "succeeded"
    ctx = jobs.JobContext(j.id)
    ctx.progress(0.2, "a")
    ctx.progress(0.205, "a", "ignored")
    assert jobs.get_runner().get(j.id).message != "ignored"
    assert jobs.get_runner().get(j.id).step == "a"


def test_cancel_between_steps(monkeypatch):
    monkeypatch.setenv("REQCLUSTER_JOB_WORKERS", "1")
    get_settings.cache_clear()
    r = jobs.get_runner()

    def fn(ctx):
        for i in range(400):
            ctx.progress(i / 400, f"s{i}")
            time.sleep(0.01)

    j = r.submit("cluster", 1, fn)
    time.sleep(0.1)
    r.cancel(j.id)
    assert _wait(r, j.id).status == "cancelled"


def test_errors_stored():
    r = jobs.get_runner()

    def bad(ctx):
        raise Invalid("bad input")

    def boom(ctx):
        raise ValueError("secret-detail")

    j = r.submit("cluster", 1, bad)
    assert (j.status, j.error, j.message) == ("failed", "bad input", "bad input")
    j = r.submit("cluster", 1, boom)
    assert j.status == "failed" and j.error == jobs.GENERIC_ERROR
    assert "secret-detail" not in (j.error + j.message)
    assert issubclass(Invalid, AppError)


def test_one_active_per_session(monkeypatch):
    monkeypatch.setenv("REQCLUSTER_JOB_WORKERS", "1")
    get_settings.cache_clear()
    r = jobs.get_runner()
    j1 = r.submit("cluster", 1, lambda ctx: time.sleep(0.3))
    with pytest.raises(Conflict):
        r.submit("recluster", 1, lambda ctx: None)
    j2 = r.submit("cluster", 2, lambda ctx: None)
    _wait(r, j1.id)
    _wait(r, j2.id)
    assert r.submit("cluster", 1, lambda ctx: None).kind == "cluster"
    assert r.latest(1, ("cluster",)) is not None and r.latest(1, ("enrich",)) is None


def test_sweep_on_startup():
    r = jobs.get_runner()
    with db.session_scope() as s:
        from reqcluster.models import JobRow

        s.add(JobRow(id="a" * 32, session_id=1, kind="cluster", status="running"))
    r.sweep_on_startup()
    j = r.get("a" * 32)
    assert j.status == "failed" and j.error == "interrupted by restart"


def test_finish_retries_locked_db(monkeypatch):
    from sqlalchemy.exc import OperationalError

    monkeypatch.setattr(jobs.time, "sleep", lambda s: None)
    real, calls = jobs.update, []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) <= 2:
            raise OperationalError("UPDATE", {}, Exception("database is locked"))
        return real(*a, **kw)

    monkeypatch.setattr(jobs, "update", flaky)
    r = jobs.get_runner()
    assert r.submit("cluster", 1, lambda ctx: None).status == "succeeded"
    assert r.submit("cluster", 1, lambda ctx: None).status == "succeeded"


def test_shutdown_nowait_drops_queued(monkeypatch):
    import threading

    monkeypatch.setenv("REQCLUSTER_JOB_WORKERS", "1")
    get_settings.cache_clear()
    r = jobs.get_runner()
    gate = threading.Event()
    j1 = r.submit("cluster", 1, lambda ctx: gate.wait(5))
    j2 = r.submit("cluster", 2, lambda ctx: None)
    jobs.reset_runner(wait=False)
    gate.set()
    _wait(r, j1.id)
    time.sleep(0.2)
    assert r.get(j2.id).status == "queued"
