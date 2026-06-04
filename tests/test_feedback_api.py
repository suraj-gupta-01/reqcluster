"""Tests for Phase 4 Human-in-the-Loop feedback API routes."""

import pytest
import sys
import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from models.database import Base, Requirement, Cluster, FeedbackCorrection, ConstraintPair, get_db
from api import routes


@pytest.fixture
def api_db(tmp_path):
    """Create an isolated FastAPI app with its own test database, following the
    same pattern used by test_enrichment_api_db.py."""
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_feedback.db'}",
        connect_args={"check_same_thread": False},
    )
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[routes.get_db] = override_get_db
    routes.pipeline_progress.clear()

    with TestClient(app) as client:
        yield client, TestingSessionLocal, engine

    routes.pipeline_progress.clear()
    engine.dispose()


def seed_session(SessionLocal):
    """Create a session with 10 requirements in 2 clusters."""
    db = SessionLocal()
    try:
        from models.database import Session as DbSessionModel

        session = DbSessionModel(
            name="feedback-test",
            filename="feedback.csv",
            status="done",
            total_requirements=10,
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        for i in range(10):
            cluster_id = 0 if i < 5 else 1
            req = Requirement(
                session_id=session.id,
                req_id=f"REQ-{i+1:03d}",
                text=f"Requirement {i+1} for thermal/power system",
                cluster_id=cluster_id,
                membership_prob=0.9,
                is_noise=False,
            )
            db.add(req)

        db.add(Cluster(session_id=session.id, cluster_id=0, label="Thermal Control", keywords=["thermal"], size=5))
        db.add(Cluster(session_id=session.id, cluster_id=1, label="Power Systems", keywords=["power"], size=5))
        db.commit()
        return session.id
    finally:
        db.close()


class TestFeedbackEndpoints:
    def test_submit_feedback_route(self, api_db):
        client, SessionLocal, _engine = api_db
        session_id = seed_session(SessionLocal)

        db = SessionLocal()
        req = db.query(Requirement).filter(Requirement.session_id == session_id).first()
        db.close()

        payload = {
            "session_id": session_id,
            "requirement_id": req.id,
            "new_cluster_id": 1,
            "confidence_score": 0.9,
            "comments": "Move requirement to Power Systems",
            "applied_by": "Senior Analyst",
        }

        res = client.post("/api/feedback/submit", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["requirement_id"] == req.id
        assert data["previous_cluster_id"] == 0
        assert data["new_cluster_id"] == 1
        assert data["status"] == "pending"

    def test_submit_feedback_not_found(self, api_db):
        client, SessionLocal, _engine = api_db
        session_id = seed_session(SessionLocal)

        payload = {
            "session_id": session_id,
            "requirement_id": 99999,  # invalid id
            "new_cluster_id": 1,
            "confidence_score": 0.9,
            "comments": "Move non-existent requirement",
            "applied_by": "Senior Analyst",
        }

        res = client.post("/api/feedback/submit", json=payload)
        assert res.status_code == 400

    def test_get_feedback_queue(self, api_db):
        client, SessionLocal, _engine = api_db
        session_id = seed_session(SessionLocal)

        db = SessionLocal()
        req = db.query(Requirement).filter(Requirement.session_id == session_id).first()
        corr = FeedbackCorrection(
            session_id=session_id,
            requirement_id=req.id,
            previous_cluster_id=0,
            new_cluster_id=1,
            confidence_score=0.95,
            comments="Test correction",
            applied_by="Analyst",
            status="pending",
        )
        db.add(corr)
        db.commit()
        db.close()

        res = client.get(f"/api/feedback/queue?session_id={session_id}")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["comments"] == "Test correction"
        assert data[0]["status"] == "pending"

    def test_review_feedback_route(self, api_db):
        client, SessionLocal, _engine = api_db
        session_id = seed_session(SessionLocal)

        db = SessionLocal()
        req = db.query(Requirement).filter(Requirement.session_id == session_id).first()
        corr = FeedbackCorrection(
            session_id=session_id,
            requirement_id=req.id,
            previous_cluster_id=0,
            new_cluster_id=1,
            confidence_score=0.95,
            comments="Test correction",
            applied_by="Analyst",
            status="pending",
        )
        db.add(corr)
        db.commit()
        feedback_id = corr.id
        db.close()

        payload = {
            "session_id": session_id,
            "feedback_id": feedback_id,
            "status": "approved",
        }

        res = client.post("/api/feedback/review", json=payload)
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "approved"

    def test_get_constraints_and_conflicts(self, api_db):
        client, SessionLocal, _engine = api_db
        session_id = seed_session(SessionLocal)

        db = SessionLocal()
        req = db.query(Requirement).filter(Requirement.session_id == session_id).first()
        corr = FeedbackCorrection(
            session_id=session_id,
            requirement_id=req.id,
            previous_cluster_id=0,
            new_cluster_id=1,
            status="pending",
        )
        db.add(corr)
        db.flush()
        # Use real requirement IDs from the seeded data
        reqs = db.query(Requirement).filter(Requirement.session_id == session_id).limit(2).all()
        cp = ConstraintPair(
            session_id=session_id,
            requirement_a_id=reqs[0].id,
            requirement_b_id=reqs[1].id,
            constraint_type="must-link",
            feedback_id=corr.id,
        )
        db.add(cp)
        db.commit()
        db.close()

        res = client.get(f"/api/feedback/constraints?session_id={session_id}")
        assert res.status_code == 200
        data = res.json()
        assert "constraint_pairs" in data
        assert len(data["constraint_pairs"]) == 1
        assert data["constraint_pairs"][0]["constraint_type"] == "must-link"
        assert "conflicts" in data
        assert data["has_conflicts"] is False

    def test_export_feedback_csv(self, api_db):
        client, SessionLocal, _engine = api_db
        session_id = seed_session(SessionLocal)

        db = SessionLocal()
        req = db.query(Requirement).filter(Requirement.session_id == session_id).first()
        corr = FeedbackCorrection(
            session_id=session_id,
            requirement_id=req.id,
            previous_cluster_id=0,
            new_cluster_id=1,
            confidence_score=0.95,
            comments="Export comment",
            applied_by="Exporter",
            status="pending",
        )
        db.add(corr)
        db.commit()
        db.close()

        res = client.get(f"/api/feedback/export?session_id={session_id}&format=csv")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/csv")
        assert "Export comment" in res.text

    def test_export_feedback_json(self, api_db):
        client, SessionLocal, _engine = api_db
        session_id = seed_session(SessionLocal)

        db = SessionLocal()
        req = db.query(Requirement).filter(Requirement.session_id == session_id).first()
        corr = FeedbackCorrection(
            session_id=session_id,
            requirement_id=req.id,
            previous_cluster_id=0,
            new_cluster_id=1,
            confidence_score=0.95,
            comments="Export comment",
            applied_by="Exporter",
            status="pending",
        )
        db.add(corr)
        db.commit()
        db.close()

        res = client.get(f"/api/feedback/export?session_id={session_id}&format=json")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("application/json")
        data = res.json()
        assert len(data) == 1
        assert data[0]["comments"] == "Export comment"
