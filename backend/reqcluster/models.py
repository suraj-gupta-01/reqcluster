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
    attributes: Mapped[dict] = mapped_column(JSON, default=dict)


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
    reports: Mapped[dict] = mapped_column(JSON, default=dict)
    n_clusters: Mapped[int] = mapped_column(Integer, default=0)
    noise_count: Mapped[int] = mapped_column(Integer, default=0)
    edit_seq: Mapped[int] = mapped_column(Integer, default=0)
    max_cluster_id: Mapped[int] = mapped_column(Integer, default=-1)  # highest id ever used
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


def _run_fk():
    return mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), index=True)


def _req_fk(**kw):
    return mapped_column(ForeignKey("requirements.id", ondelete="CASCADE"), **kw)


class EnrichmentRow(Base):
    """Session-scoped: an LLM expansion of one requirement text survives re-clustering."""
    __tablename__ = "enrichments"
    __table_args__ = (UniqueConstraint("requirement_id", "provider", "prompt_version"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), index=True
    )
    requirement_id: Mapped[int] = _req_fk()
    provider: Mapped[str] = mapped_column(String)
    prompt_version: Mapped[str] = mapped_column(String)
    text_hash: Mapped[str] = mapped_column(String)  # sha256 of the source text; stale if changed
    status: Mapped[str] = mapped_column(String)  # succeeded | failed
    source: Mapped[str] = mapped_column(String)  # llm | cache | fallback
    expanded_text: Mapped[str] = mapped_column(Text, default="")
    domain_terms: Mapped[list] = mapped_column(JSON, default=list)
    functional_intent: Mapped[str] = mapped_column(Text, default="")
    mentioned_components: Mapped[list] = mapped_column(JSON, default=list)
    assumptions: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    warnings: Mapped[list] = mapped_column(JSON, default=list)
    quality: Mapped[dict] = mapped_column(JSON, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = _ts(default=_now)


# Everything below is run-scoped: a new run (re-cluster) makes it unreachable, and run
# pruning deletes it through the FK cascade. This is what the acceptance suite requires.

class SuggestionRow(Base):
    __tablename__ = "suggestions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = _run_fk()
    edit_seq: Mapped[int] = mapped_column(Integer)  # run.edit_seq when generated; mismatch = stale
    kind: Mapped[str] = mapped_column(String)  # merge | split
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|accepted|rejected
    payload: Mapped[dict] = mapped_column(JSON, default=dict)  # cluster ids, scores, proposal
    rationale: Mapped[str] = mapped_column(Text, default="")
    provider: Mapped[str] = mapped_column(String, default="mock")
    created_at: Mapped[datetime] = _ts(default=_now)
    decided_at: Mapped[datetime | None] = _ts(nullable=True)


class FeedbackRow(Base):
    __tablename__ = "feedback"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = _run_fk()
    requirement_id: Mapped[int] = _req_fk(index=True)
    old_cluster_id: Mapped[int] = mapped_column(Integer)
    new_cluster_id: Mapped[int] = mapped_column(Integer)  # -1 = noise
    status: Mapped[str] = mapped_column(String, default="pending")  # pending|approved|rejected
    comments: Mapped[str] = mapped_column(Text, default="")
    applied_by: Mapped[str] = mapped_column(String, default="")
    created_at: Mapped[datetime] = _ts(default=_now)
    reviewed_at: Mapped[datetime | None] = _ts(nullable=True)


class ConstraintRow(Base):
    __tablename__ = "constraints"
    __table_args__ = (UniqueConstraint("run_id", "a_id", "b_id", "kind"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = _run_fk()
    a_id: Mapped[int] = _req_fk()  # requirement pk, always a_id < b_id
    b_id: Mapped[int] = _req_fk()
    kind: Mapped[str] = mapped_column(String)  # must | cannot
    feedback_id: Mapped[int | None] = mapped_column(
        ForeignKey("feedback.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = _ts(default=_now)


class AuditRow(Base):
    __tablename__ = "audit"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = _run_fk()
    action: Mapped[str] = mapped_column(String)
    actor: Mapped[str] = mapped_column(String, default="")
    details: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = _ts(default=_now)


class DependencyTreeRow(Base):
    __tablename__ = "dependency_trees"
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    params: Mapped[dict] = mapped_column(JSON, default=dict)
    nodes: Mapped[list] = mapped_column(JSON, default=list)
    edges: Mapped[list] = mapped_column(JSON, default=list)  # source/target are requirement pks
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    rationale: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = _ts(default=_now)


class DependencyExplanationRow(Base):
    __tablename__ = "dependency_explanations"
    run_id: Mapped[int] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), primary_key=True)
    source: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    target: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=False)
    provider: Mapped[str] = mapped_column(String)
    explanation: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = _ts(default=_now)


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
