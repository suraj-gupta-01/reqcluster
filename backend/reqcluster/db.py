"""Lazy SQLAlchemy engine and session helpers."""
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

from reqcluster.config import get_settings

_lock = threading.Lock()
_engine: Engine | None = None


def _build(url: str) -> Engine:
    if not url.startswith("sqlite"):
        return create_engine(url, pool_pre_ping=True)
    path = url.split("///", 1)[-1]
    if path and not path.startswith(":memory:"):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    # NullPool: one short-lived connection per use; WAL lets readers coexist with a writer.
    engine = create_engine(url, connect_args={"check_same_thread": False}, poolclass=NullPool)

    @event.listens_for(engine, "connect")
    def _pragmas(conn, _record):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.close()

    return engine


def get_engine() -> Engine:
    global _engine
    with _lock:
        if _engine is None:
            _engine = _build(get_settings().database_url)
        return _engine


def configure(url: str) -> Engine:
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = _build(url)
        return _engine


def reset() -> None:
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None


def SessionLocal() -> Session:  # noqa: N802 - sessionmaker-style name, bound lazily
    return Session(get_engine(), expire_on_commit=False, autoflush=False)


@contextmanager
def session_scope() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
        s.commit()
    except BaseException:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    s = SessionLocal()
    try:
        yield s
    finally:
        s.close()


def init_db() -> None:
    from reqcluster.models import Base

    Base.metadata.create_all(get_engine())
