"""Tests for Phase 4 Human-in-the-Loop feedback service layer."""

import pytest
import numpy as np
import sys
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from models.database import Base, Requirement, Cluster, FeedbackCorrection, ConstraintPair
from services.feedback_service import submit_feedback, review_feedback, get_feedback_queue
from core.feedback_bridge import detect_constraints_conflicts, generate_constraints_for_correction
from llm_services.feedback_analyst import FeedbackAnalyst

# In-memory test database
TEST_DATABASE_URL = "sqlite:///./test_feedback_service.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def mock_session_data(db_session):
    # Setup test requirements and clusters
    req1 = Requirement(
        session_id=1,
        req_id="REQ-001",
        text="The system shall display system temperature.",
        cluster_id=0,
        membership_prob=0.95,
        is_noise=False,
    )
    req2 = Requirement(
        session_id=1,
        req_id="REQ-002",
        text="The system shall support active cooling loop.",
        cluster_id=0,
        membership_prob=0.88,
        is_noise=False,
    )
    req3 = Requirement(
        session_id=1,
        req_id="REQ-003",
        text="The system shall regulate battery voltage.",
        cluster_id=1,
        membership_prob=0.92,
        is_noise=False,
    )
    req4 = Requirement(
        session_id=1,
        req_id="REQ-004",
        text="The system shall charge battery cell stacks.",
        cluster_id=1,
        membership_prob=0.91,
        is_noise=False,
    )
    db_session.add_all([req1, req2, req3, req4])

    c0 = Cluster(
        session_id=1,
        cluster_id=0,
        label="Thermal Control",
        keywords=["cooling", "temperature", "loop"],
        size=2,
    )
    c1 = Cluster(
        session_id=1,
        cluster_id=1,
        label="Power Management",
        keywords=["battery", "voltage", "charge"],
        size=2,
    )
    db_session.add_all([c0, c1])
    db_session.commit()
    return 1


class DummyRequest:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def test_submit_feedback_mutation(db_session, mock_session_data):
    # Move REQ-001 (id=1, old_cluster=0) to cluster 1
    req = db_session.query(Requirement).filter(Requirement.req_id == "REQ-001").first()
    assert req.cluster_id == 0

    request = DummyRequest(
        session_id=1,
        requirement_id=req.id,
        new_cluster_id=1,
        confidence_score=0.95,
        comments="Voltage or power related",
        applied_by="Test Analyst",
    )

    corr = submit_feedback(db_session, request)

    assert corr.id is not None
    assert corr.requirement_id == req.id
    assert corr.previous_cluster_id == 0
    assert corr.new_cluster_id == 1
    assert corr.confidence_score == 0.95
    assert corr.status == "pending"

    # Verify requirement assignment has changed immediately in the DB
    db_session.refresh(req)
    assert req.cluster_id == 1

    # Verify cluster sizes were mutated
    c0 = db_session.query(Cluster).filter(Cluster.cluster_id == 0).first()
    c1 = db_session.query(Cluster).filter(Cluster.cluster_id == 1).first()
    assert c0.size == 1
    assert c1.size == 3


def test_submit_feedback_raises_identical_cluster(db_session, mock_session_data):
    req = db_session.query(Requirement).filter(Requirement.req_id == "REQ-001").first()
    request = DummyRequest(
        session_id=1,
        requirement_id=req.id,
        new_cluster_id=0,  # Identical to current
        confidence_score=1.0,
        comments="No change",
        applied_by="Analyst",
    )
    with pytest.raises(ValueError, match="assignment is identical"):
        submit_feedback(db_session, request)


def test_submit_feedback_dynamic_confidence(db_session, mock_session_data):
    req = db_session.query(Requirement).filter(Requirement.req_id == "REQ-001").first()
    # When confidence is None, dynamic confidence calculation runs
    request = DummyRequest(
        session_id=1,
        requirement_id=req.id,
        new_cluster_id=1,  # Target cluster 1 ("Power Management", keywords: ["battery", "voltage", "charge"])
        confidence_score=None,
        comments="Move to power battery and voltage charging systems",
        applied_by="Test Analyst",
    )
    corr = submit_feedback(db_session, request)
    # Overlap with "battery", "voltage", "charge", "power" should yield high confidence
    assert corr.confidence_score > 0.5


def test_review_feedback_approve(db_session, mock_session_data):
    req = db_session.query(Requirement).filter(Requirement.req_id == "REQ-001").first()
    request = DummyRequest(
        session_id=1,
        requirement_id=req.id,
        new_cluster_id=1,
        confidence_score=0.9,
        comments="Related to battery",
        applied_by="Test Analyst",
    )
    corr = submit_feedback(db_session, request)
    assert corr.status == "pending"

    reviewed = review_feedback(db_session, 1, corr.id, "approved")
    assert reviewed.status == "approved"

    # Verify state remains correct
    db_session.refresh(req)
    assert req.cluster_id == 1


def test_review_feedback_reject_rollback(db_session, mock_session_data):
    req = db_session.query(Requirement).filter(Requirement.req_id == "REQ-001").first()
    request = DummyRequest(
        session_id=1,
        requirement_id=req.id,
        new_cluster_id=1,
        confidence_score=0.8,
        comments="Battery management",
        applied_by="Test Analyst",
    )
    corr = submit_feedback(db_session, request)
    
    # Confirm constraint pairs were created
    cps = db_session.query(ConstraintPair).filter(ConstraintPair.feedback_id == corr.id).all()
    assert len(cps) > 0

    # Reject the feedback correction
    reviewed = review_feedback(db_session, 1, corr.id, "rejected")
    assert reviewed.status == "rejected"

    # Verify requirement assignment is rolled back
    db_session.refresh(req)
    assert req.cluster_id == 0

    # Verify cluster sizes are reverted
    c0 = db_session.query(Cluster).filter(Cluster.cluster_id == 0).first()
    c1 = db_session.query(Cluster).filter(Cluster.cluster_id == 1).first()
    assert c0.size == 2
    assert c1.size == 2

    # Verify constraint pairs were deleted
    cps_after = db_session.query(ConstraintPair).filter(ConstraintPair.feedback_id == corr.id).all()
    assert len(cps_after) == 0


def test_detect_constraint_conflicts(db_session, mock_session_data):
    # Setup transitive conflict:
    # REQ-001 must-link REQ-002 (feedback 10)
    # REQ-002 must-link REQ-003 (feedback 11)
    # REQ-001 cannot-link REQ-003 (feedback 12)
    # Let's write them directly to the DB to bypass representative extractor logic for simplicity.
    corr1 = FeedbackCorrection(session_id=1, requirement_id=1, new_cluster_id=0, status="approved")
    corr2 = FeedbackCorrection(session_id=1, requirement_id=2, new_cluster_id=0, status="approved")
    corr3 = FeedbackCorrection(session_id=1, requirement_id=1, new_cluster_id=1, status="approved")
    db_session.add_all([corr1, corr2, corr3])
    db_session.flush()

    cp1 = ConstraintPair(session_id=1, requirement_a_id=1, requirement_b_id=2, constraint_type="must-link", feedback_id=corr1.id)
    cp2 = ConstraintPair(session_id=1, requirement_a_id=2, requirement_b_id=3, constraint_type="must-link", feedback_id=corr2.id)
    cp3 = ConstraintPair(session_id=1, requirement_a_id=1, requirement_b_id=3, constraint_type="cannot-link", feedback_id=corr3.id)
    db_session.add_all([cp1, cp2, cp3])
    db_session.commit()

    conflicts = detect_constraints_conflicts(db_session, 1)
    assert len(conflicts) == 1
    assert "REQ-001" in conflicts[0]["message"] or "1" in conflicts[0]["message"]
    assert "REQ-003" in conflicts[0]["message"] or "3" in conflicts[0]["message"]
