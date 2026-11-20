"""Background job runner backed by the jobs table."""
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from reqcluster import runtime
from reqcluster.config import Settings, get_settings
from reqcluster.db import session_scope
from reqcluster.domain import Job
from reqcluster.errors import AppError, Cancelled, Conflict, NotFound
from reqcluster.models import JobRow, to_job

log = logging.getLogger(__name__)
GENERIC_ERROR = "The job failed because of an internal error."


def _now() -> datetime:
    return datetime.now(UTC)


class JobContext:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self._frac = -1.0
        self._step = None
        self._t = 0.0

    def check_cancel(self) -> None:
        with session_scope() as s:
            row = s.get(JobRow, self.job_id)
            if row is not None and row.cancel_requested:
                raise Cancelled("job cancelled")

    def progress(self, fraction: float, step: str, message: str | None = None) -> None:
        now = time.monotonic()
        if (
            step == self._step
            and abs(fraction - self._frac) < 0.01
            and now - self._t < 0.5
        ):
            return
        self._frac, self._step, self._t = fraction, step, now
        with session_scope() as s:
            row = s.get(JobRow, self.job_id)
            if row is None:
                return
            row.progress, row.step, row.message = fraction, step, message
            if row.cancel_requested:
                s.commit()
                raise Cancelled("job cancelled")


class JobRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._cpu = self._io = None
        if settings.job_workers > 0:
            kw = {"initializer": runtime.configure_process, "initargs": (settings,)}
            self._cpu = ThreadPoolExecutor(max(1, settings.job_workers), "job-cpu", **kw)
            self._io = ThreadPoolExecutor(max(1, settings.llm_concurrency), "job-io", **kw)

    def submit(
        self,
        kind: str,
        session_id: int,
        fn: Callable[[JobContext], Any],
        params: dict | None = None,
    ) -> Job:
        try:
            with session_scope() as s:
                row = JobRow(
                    session_id=session_id,
                    kind=kind,
                    status="queued",
                    params=params or {},
                )
                s.add(row)
                s.flush()
                job_id = row.id
        except IntegrityError as exc:
            msg = str(exc).upper()
            if "FOREIGN KEY" in msg:  # session deleted concurrently
                raise NotFound(f"session {session_id} not found") from exc
            if "uq_jobs_one_active" in str(exc) or "UNIQUE" in msg:
                raise Conflict("another job is already queued or running for this session") from exc
            raise
        if self._cpu is None:
            self._execute(job_id, fn)
            return self.get(job_id)
        job = self.get(job_id)
        (self._io if kind == "enrich" else self._cpu).submit(self._execute, job_id, fn)
        return job

    def _finish(self, job_id: str, **values: Any) -> None:
        # A finished job must not stay 'running' (it would block the session): retry until the
        # write lands. Process death is covered by the startup sweep.
        delay, attempt = 0.2, 0
        while True:
            attempt += 1
            try:
                with session_scope() as s:
                    s.execute(update(JobRow).where(JobRow.id == job_id, JobRow.status == "running")
                              .values(finished_at=_now(), **values))
                return
            except Exception:
                log.warning("job %s: status write failed (attempt %d)", job_id, attempt, exc_info=True)
                time.sleep(delay)
                delay = min(delay * 2, 1.0)

    def _execute(self, job_id: str, fn: Callable[[JobContext], Any]) -> None:
        try:
            with session_scope() as s:
                started = s.execute(
                    update(JobRow).where(JobRow.id == job_id, JobRow.status == "queued")
                    .values(status="running", started_at=_now())
                ).rowcount
        except Exception:
            log.exception("job %s: start write failed", job_id)
            with session_scope() as s:
                s.execute(update(JobRow).where(JobRow.id == job_id, JobRow.status == "queued")
                          .values(status="failed", error=GENERIC_ERROR, message=GENERIC_ERROR,
                                  finished_at=_now()))
            return
        if not started:
            return
        try:
            r = fn(JobContext(job_id))
            result = r.model_dump(mode="json") if isinstance(r, BaseModel) else r
            if result is not None and not isinstance(result, dict):
                result = None
            self._finish(job_id, status="succeeded", progress=1.0, result=result)
        except Cancelled:
            self._finish(job_id, status="cancelled")
        except AppError as exc:
            self._finish(job_id, status="failed", error=str(exc), message=str(exc))
        except Exception:
            log.exception("job %s failed", job_id)
            self._finish(job_id, status="failed", error=GENERIC_ERROR, message=GENERIC_ERROR)

    def get(self, job_id: str) -> Job:
        with session_scope() as s:
            row = s.get(JobRow, job_id)
            if row is None:
                raise NotFound(f"job {job_id} not found")
            return to_job(row)

    def latest(self, session_id: int, kinds: tuple[str, ...]) -> Job | None:
        with session_scope() as s:
            row = s.scalars(
                select(JobRow)
                .where(JobRow.session_id == session_id, JobRow.kind.in_(kinds))
                .order_by(JobRow.created_at.desc())
                .limit(1)
            ).first()
            return to_job(row) if row else None

    def cancel(self, job_id: str) -> Job:
        with session_scope() as s:
            if s.get(JobRow, job_id) is None:
                raise NotFound(f"job {job_id} not found")
            if not s.execute(
                update(JobRow).where(JobRow.id == job_id, JobRow.status == "queued")
                .values(status="cancelled", finished_at=_now())
            ).rowcount:
                s.execute(update(JobRow).where(JobRow.id == job_id, JobRow.status == "running")
                          .values(cancel_requested=True))
        return self.get(job_id)

    def sweep_on_startup(self) -> None:
        with session_scope() as s:
            s.execute(
                update(JobRow)
                .where(JobRow.status.in_(("queued", "running")))
                .values(
                    status="failed",
                    error="interrupted by restart",
                    message="interrupted by restart",
                    finished_at=_now(),
                )
            )

    def shutdown(self, wait: bool = True) -> None:
        for ex in (self._cpu, self._io):
            if ex is not None:
                ex.shutdown(wait=wait, cancel_futures=not wait)


_runner: JobRunner | None = None
_lock = threading.Lock()


def get_runner() -> JobRunner:
    global _runner
    with _lock:
        if _runner is None:
            _runner = JobRunner(get_settings())
        return _runner


def reset_runner(wait: bool = True) -> None:
    global _runner
    with _lock:
        r, _runner = _runner, None
    if r is not None:
        r.shutdown(wait=wait)
