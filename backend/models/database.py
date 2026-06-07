"""
Database configuration and ORM models for ReqCluster.

Engine selection
----------------
Set DATABASE_URL in your environment to use PostgreSQL (recommended for
production and any deployment with more than ~5 concurrent users).

    DATABASE_URL=postgresql://user:password@localhost:5432/reqcluster

If DATABASE_URL is **not** set, the engine falls back to a local SQLite file
(data/reqcluster.db).  A startup warning is emitted so you are always aware of
which backend is active — SQLite is intentional only for quick local dev.

PostgreSQL is strictly required once you exceed ~20 K requirements or need
concurrent write throughput (bottleneck #1 in SCALABILITY_ROADMAP.md).

Index strategy
--------------
Indexes are chosen based on actual hot query paths (see routes.py).  Each index
is annotated with the query it accelerates and the write-overhead trade-off.
Indexes on low-cardinality boolean columns (is_noise) are included only where
the filter dramatically reduces result set size.
"""

import logging
import os
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import NullPool, QueuePool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Timestamp helper
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    """Naive UTC timestamp.

    Replaces the deprecated ``datetime.utcnow`` (removed in a future Python)
    while preserving the existing naive-UTC storage semantics so persisted
    timestamps stay comparable with historical rows.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ---------------------------------------------------------------------------
# Engine / URL selection
# ---------------------------------------------------------------------------

_DATABASE_URL: str | None = os.getenv("DATABASE_URL")

if _DATABASE_URL:
    # ── PostgreSQL (or any other SQLAlchemy-supported RDBMS) ─────────────────
    # Mask the password in log output.
    _safe_url = _DATABASE_URL
    if "@" in _DATABASE_URL:
        scheme, rest = _DATABASE_URL.split("://", 1)
        userinfo, hostpart = rest.split("@", 1)
        _safe_url = f"{scheme}://***@{hostpart}"

    logger.info(
        "ReqCluster: using PostgreSQL database → %s  "
        "(set DATABASE_URL='' to switch back to SQLite)",
        _safe_url,
    )

    engine = create_engine(
        _DATABASE_URL,
        # ------------------------------------------------------------------
        # QueuePool: maintains a pool of persistent connections so each
        # request doesn't pay the TCP handshake cost.
        #   pool_size=20   — base connections held open at all times
        #   max_overflow=10 — extra connections allowed under peak load
        #   pool_pre_ping   — discard stale connections silently
        # ------------------------------------------------------------------
        poolclass=QueuePool,
        pool_size=20,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=3600,  # recycle connections every hour to avoid server-side timeouts
    )

else:
    # ── SQLite fallback (local dev only) ─────────────────────────────────────
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DB_PATH = os.path.join(BASE_DIR, "..", "data", "reqcluster.db")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _DATABASE_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

    logger.warning(
        "\n"
        "╔══════════════════════════════════════════════════════════════════╗\n"
        "║  ⚠  ReqCluster is running with SQLite (local dev mode)          ║\n"
        "║                                                                  ║\n"
        "║  SQLite does NOT support concurrent writes and will become a     ║\n"
        "║  bottleneck beyond ~5 users or ~20 K requirements.              ║\n"
        "║                                                                  ║\n"
        "║  To switch to PostgreSQL set DATABASE_URL in your .env:         ║\n"
        "║    DATABASE_URL=postgresql://user:pass@localhost:5432/reqcluster ║\n"
        "║                                                                  ║\n"
        "║  If you intentionally want SQLite, you can silence this warning  ║\n"
        "║  by setting REQCLUSTER_SQLITE_OK=1 in your environment.         ║\n"
        "╚══════════════════════════════════════════════════════════════════╝"
    )

    engine = create_engine(
        _DATABASE_URL,
        # check_same_thread=False: connections may be used across FastAPI's
        # threadpool. timeout: wait (don't error) when the DB is briefly locked.
        connect_args={"check_same_thread": False, "timeout": 30},
        # NullPool: each request gets its OWN short-lived connection. A shared
        # connection (StaticPool) breaks under concurrency - one request's
        # commit/rollback invalidates another request's open cursor
        # ("Cursor needed to be reset ..."). WAL mode (below) lets these
        # per-request connections read concurrently while writes serialize.
        poolclass=NullPool,
    )

    # Enable WAL mode so reads don't block writes even under SQLite.
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragmas(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=-64000")  # 64 MB page cache
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ---------------------------------------------------------------------------
# ORM Models
# ---------------------------------------------------------------------------

class Session(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=utcnow)
    status = Column(String, default="uploaded")  # uploaded, processing, done, error
    total_requirements = Column(Integer, default=0)
    total_clusters = Column(Integer, default=0)
    noise_count = Column(Integer, default=0)

    __table_args__ = (
        # Hot path: GET /sessions lists sessions ordered by created_at desc,
        # filtered (optionally) by status.  Composite (status, created_at)
        # lets the DB skip a sort on the most common "all statuses" list.
        # Write trade-off: negligible — sessions are inserted once and rarely
        # updated, so index maintenance is essentially free.
        Index("ix_session_status_created", "status", "created_at"),
    )


class Requirement(Base):
    __tablename__ = "requirements"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    req_id = Column(String, nullable=True)
    text = Column(Text, nullable=False)
    module = Column(String, nullable=True)
    section = Column(String, nullable=True)
    cluster_id = Column(Integer, nullable=True)
    membership_prob = Column(Float, nullable=True)
    umap_x = Column(Float, nullable=True)
    umap_y = Column(Float, nullable=True)
    is_noise = Column(Boolean, default=False)

    __table_args__ = (
        # ── (session_id, cluster_id) ────────────────────────────────────────
        # Hot path: GET /requirements?session_id=X&cluster_id=Y
        #           GET /cluster/<id>?session_id=X  (inner join on cluster_id)
        # Both filters are applied on EVERY cluster-detail page load.
        # Write trade-off: moderate — requirements are bulk-inserted once per
        # pipeline run and then only updated in-place (cluster_id, umap_x/y).
        # The index pays for itself after the first query on any dataset > 1 K.
        Index("ix_req_session_cluster", "session_id", "cluster_id"),
        #
        # ── (session_id, is_noise) ──────────────────────────────────────────
        # Hot path: noise-point filtering on scatter plot and export.
        # is_noise is low-cardinality (true/false) but combined with
        # session_id the index is selective enough to be useful.
        # Write trade-off: same as above — small, one-time cost per session.
        Index("ix_req_session_noise", "session_id", "is_noise"),
        #
        # NOTE: A separate (session_id, module/section) index was considered
        # but the current API doesn't filter by module/section at the DB
        # level, so it would be write-overhead with no read benefit.
    )


class Cluster(Base):
    __tablename__ = "clusters"

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    cluster_id = Column(Integer, nullable=False)
    label = Column(String, nullable=False)
    keywords = Column(JSON, nullable=True)
    size = Column(Integer, default=0)

    __table_args__ = (
        # Hot path: GET /cluster/<cluster_id>?session_id=X filters on both
        # session_id AND cluster_id.  Without this index the DB scans all
        # clusters for the session.
        # Write trade-off: clusters are written once per pipeline and never
        # updated; index maintenance cost is zero after creation.
        Index("ix_cluster_session_cluster_id", "session_id", "cluster_id"),
    )


class Graph(Base):
    __tablename__ = "graphs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, unique=True)
    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)

    # session_id already has a UNIQUE constraint (→ implicit index).
    # No additional indexes needed here.


class DependencyTree(Base):
    __tablename__ = "dependency_trees"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, unique=True, index=True)
    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)
    stats = Column(JSON, nullable=True)
    rationale = Column(JSON, nullable=True)  # per-cluster + per-edge rationale document
    provider = Column(String(80), nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # session_id UNIQUE gives a free index; no additional index warranted.


class EnrichedRequirement(Base):
    __tablename__ = "enriched_requirements"
    __table_args__ = (
        UniqueConstraint(
            "session_id",
            "requirement_db_id",
            "requirement_text_hash",
            "provider",
            "model",
            "prompt_version",
            name="uq_enriched_requirement_identity",
        ),
        # Hot path: enrichment status polling and result listing both filter on
        # (session_id, requirement_db_id) or (session_id, status).
        # These were already present before — kept as-is.
        Index("ix_enriched_session_requirement", "session_id", "requirement_db_id"),
        Index("ix_enriched_session_status", "session_id", "status"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    requirement_db_id = Column(Integer, nullable=False, index=True)
    requirement_id = Column(String, nullable=True, index=True)
    requirement_text_hash = Column(String(64), nullable=False, index=True)
    provider = Column(String(80), nullable=False, index=True)
    model = Column(String(160), nullable=False)
    prompt_version = Column(String(120), nullable=False)
    embedding_mode_recommended = Column(String(20), default="hybrid")
    expanded_text = Column(Text, nullable=True)
    domain_terms_json = Column(JSON, nullable=True)
    functional_intent = Column(Text, nullable=True)
    mentioned_components_json = Column(JSON, nullable=True)
    assumptions_json = Column(JSON, nullable=True)
    confidence = Column(Float, nullable=True)
    warnings_json = Column(JSON, nullable=True)
    quality_report_json = Column(JSON, nullable=True)
    status = Column(String(20), default="pending", index=True)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class RefinementSuggestion(Base):
    __tablename__ = "refinement_suggestions"
    __table_args__ = (
        # Both indexes already target hot query paths — kept unchanged.
        Index("ix_refinement_session_status", "session_id", "status"),
        Index("ix_refinement_session_type", "session_id", "suggestion_type"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    suggestion_type = Column(String(10), nullable=False)  # merge / split
    status = Column(String(20), default="pending", index=True)  # pending / accepted / rejected / applied

    # Merge fields
    cluster_a_id = Column(Integer, nullable=True)
    cluster_b_id = Column(Integer, nullable=True)
    # Split fields
    cluster_id = Column(Integer, nullable=True)

    # Scores
    similarity_score = Column(Float, nullable=True)
    silhouette_delta = Column(Float, nullable=True)
    coherence_score = Column(Float, nullable=True)
    spread_score = Column(Float, nullable=True)
    bimodality_score = Column(Float, nullable=True)

    # Text
    rationale = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    cluster_a_label = Column(String, nullable=True)
    cluster_b_label = Column(String, nullable=True)
    cluster_label = Column(String, nullable=True)

    # JSON
    representative_req_ids_json = Column(JSON, nullable=True)
    sub_labels_json = Column(JSON, nullable=True)
    sub_cluster_sizes_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
    applied_at = Column(DateTime, nullable=True)


class RefinementAuditLog(Base):
    __tablename__ = "refinement_audit_log"
    __table_args__ = (
        Index("ix_audit_session", "session_id"),
        Index("ix_audit_suggestion", "suggestion_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    suggestion_id = Column(Integer, nullable=False, index=True)
    action = Column(String(20), nullable=False)  # applied / rejected / reverted
    before_state_json = Column(JSON, nullable=True)
    after_state_json = Column(JSON, nullable=True)
    applied_by = Column(String, nullable=True)  # placeholder for Phase 4 user identity
    created_at = Column(DateTime, default=utcnow)


class FeedbackCorrection(Base):
    __tablename__ = "feedback_corrections"
    __table_args__ = (
        Index("ix_feedback_session_status", "session_id", "status"),
        Index("ix_feedback_session_requirement", "session_id", "requirement_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    requirement_id = Column(Integer, nullable=False, index=True)
    previous_cluster_id = Column(Integer, nullable=True)
    new_cluster_id = Column(Integer, nullable=True)
    confidence_score = Column(Float, nullable=False, default=1.0)
    comments = Column(Text, nullable=True)
    applied_by = Column(String, nullable=True)
    status = Column(String(20), default="pending", index=True)  # pending, approved, rejected
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)


class ClusteringIteration(Base):
    __tablename__ = "clustering_iterations"
    __table_args__ = (
        # Hot path: GET /quality/history?session_id=X orders by iteration asc.
        # (session_id, iteration) lets the DB satisfy ORDER BY without a sort.
        # Write trade-off: iterations are appended (never updated) so the index
        # is maintained by simple B-tree appends — essentially zero cost.
        Index("ix_iteration_session_iter", "session_id", "iteration"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    iteration = Column(Integer, nullable=False, default=0)
    n_clusters = Column(Integer, nullable=True)
    noise_count = Column(Integer, nullable=True)
    noise_rate = Column(Float, nullable=True)
    silhouette = Column(Float, nullable=True)
    must_link_pairs = Column(Integer, default=0)
    cannot_link_pairs = Column(Integer, default=0)
    points_moved = Column(Integer, default=0)
    created_at = Column(DateTime, default=utcnow)


class ConstraintPair(Base):
    __tablename__ = "constraint_pairs"
    __table_args__ = (
        # Hot path: GET /feedback/constraints filters on (session_id) and joins
        # FeedbackCorrection — the existing ix_constraint_session index covers
        # this adequately.  A (session_id, constraint_type) composite was
        # evaluated but the current API doesn't filter by type alone, so the
        # extra index would add write cost with no measurable read gain.
        Index("ix_constraint_session", "session_id"),
        Index("ix_constraint_feedback", "feedback_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    requirement_a_id = Column(Integer, nullable=False, index=True)
    requirement_b_id = Column(Integer, nullable=False, index=True)
    constraint_type = Column(String(20), nullable=False)  # must-link / cannot-link
    feedback_id = Column(Integer, nullable=False, index=True)  # Links to FeedbackCorrection
    created_at = Column(DateTime, default=utcnow)


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    Base.metadata.create_all(bind=engine)


def reset_stale_sessions():
    """Mark any session left in 'processing' (e.g. from a crash/restart) as
    'error' so it isn't permanently blocked by the in-progress guard."""
    db = SessionLocal()
    try:
        stale = db.query(Session).filter(Session.status == "processing").all()
        for s in stale:
            s.status = "error"
        if stale:
            db.commit()
            logger.warning(
                "reset_stale_sessions: marked %d session(s) as 'error'", len(stale)
            )
    finally:
        db.close()
