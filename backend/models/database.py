from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    JSON,
    Boolean,
    UniqueConstraint,
    Index,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "..", "data", "reqcluster.db")
os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

DATABASE_URL = f"sqlite:///{os.path.abspath(DB_PATH)}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    filename = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="uploaded")  # uploaded, processing, done, error
    total_requirements = Column(Integer, default=0)
    total_clusters = Column(Integer, default=0)
    noise_count = Column(Integer, default=0)


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


class Cluster(Base):
    __tablename__ = "clusters"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    cluster_id = Column(Integer, nullable=False)
    label = Column(String, nullable=False)
    keywords = Column(JSON, nullable=True)
    size = Column(Integer, default=0)


class Graph(Base):
    __tablename__ = "graphs"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, unique=True)
    nodes = Column(JSON, nullable=False)
    edges = Column(JSON, nullable=False)


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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class RefinementSuggestion(Base):
    __tablename__ = "refinement_suggestions"
    __table_args__ = (
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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
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
    created_at = Column(DateTime, default=datetime.utcnow)


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
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ConstraintPair(Base):
    __tablename__ = "constraint_pairs"
    __table_args__ = (
        Index("ix_constraint_session", "session_id"),
        Index("ix_constraint_feedback", "feedback_id"),
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    requirement_a_id = Column(Integer, nullable=False, index=True)
    requirement_b_id = Column(Integer, nullable=False, index=True)
    constraint_type = Column(String(20), nullable=False)  # must-link / cannot-link
    feedback_id = Column(Integer, nullable=False, index=True)  # Links to FeedbackCorrection
    created_at = Column(DateTime, default=datetime.utcnow)


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
    finally:
        db.close()
