"""ORM tables."""
import uuid
from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from reqcluster.domain import Job


def _now() -> datetime:
    return datetime.now(UTC)


def new_job_id() -> str:
    return uuid.uuid4().hex


def _ts(**kw):
    return mapped_column(DateTime(timezone=True), **kw)


class Base(DeclarativeBase):
    pass


class SessionRow(Base):
    __tablename__ = "sessions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = _ts(default=_now)
    n_requirements: Mapped[int] = mapped_column(Integer, default=0)
    current_run_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class RequirementRow(Base):
    __tablename__ = "requirements"
    __table_args__ = (UniqueConstraint("session_id", "ordinal"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    ordinal: Mapped[int] = mapped_column(Integer)
    req_id: Mapped[str] = mapped_column(String)
    text: Mapped[str] = mapped_column(Text)


class RunRow(Base):
    __tablename__ = "runs"
    __table_args__ = (UniqueConstraint("session_id", "revision"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    revision: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String)  # "cluster" | "constrained"
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    durations: Mapped[dict] = mapped_column(JSON, default=dict)
    n_clusters: Mapped[int] = mapped_column(Integer, default=0)
    noise_count: Mapped[int] = mapped_column(Integer, default=0)
    edit_seq: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = _ts(default=_now)


class AssignmentRow(Base):
    __tablename__ = "assignments"
    __table_args__ = (Index("ix_assignments_run_cluster", "run_id", "cluster_id"),)
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    requirement_id: Mapped[int] = mapped_column(
        ForeignKey("requirements.id", ondelete="CASCADE"), primary_key=True
    )
    cluster_id: Mapped[int] = mapped_column(Integer)  # -1 = noise
    probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    x: Mapped[float | None] = mapped_column(Float, nullable=True)
    y: Mapped[float | None] = mapped_column(Float, nullable=True)


class ClusterRow(Base):
    __tablename__ = "clusters"
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    cluster_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    size: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String)
    keywords: Mapped[list] = mapped_column(JSON, default=list)


class GraphRow(Base):
    __tablename__ = "graphs"
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    edges: Mapped[list] = mapped_column(JSON, default=list)  # [[req_pk_i, req_pk_j, sim], ...]


class JobRow(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index(
            "uq_jobs_one_active",
            "session_id",
            unique=True,
            sqlite_where=text("status IN ('queued','running')"),
            postgresql_where=text("status IN ('queued','running')"),
        ),
    )
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_job_id)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String)
    status: Mapped[str] = mapped_column(String)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    step: Mapped[str] = mapped_column(String, default="")
    message: Mapped[str | None] = mapped_column(String, nullable=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = _ts(default=_now)
    started_at: Mapped[datetime | None] = _ts(nullable=True)
    finished_at: Mapped[datetime | None] = _ts(nullable=True)


def to_job(row: JobRow) -> Job:
    return Job(
        id=row.id,
        session_id=row.session_id,
        kind=row.kind,
        status=row.status,
        progress=row.progress,
        step=row.step,
        message=row.message,
        result=row.result,
        error=row.error,
        created_at=row.created_at,
        started_at=row.started_at,
        finished_at=row.finished_at,
    )
