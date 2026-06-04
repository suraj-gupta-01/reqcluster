"""Tests for Phase 3 refinement API endpoints."""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import Base, Session, Requirement, Cluster, RefinementSuggestion, RefinementAuditLog, get_db
from api.routes import get_db as route_get_db
from main import app


# In-memory test database
TEST_DATABASE_URL = "sqlite:///./test_refinement.db"
engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db
app.dependency_overrides[route_get_db] = override_get_db


@pytest.fixture(autouse=True)
def setup_db():
    """Create tables before each test, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def populated_session():
    """Create a session with requirements and clusters for testing."""
    db = TestSessionLocal()
    try:
        session = Session(
            name="Test Session",
            filename="test.csv",
            status="done",
            total_requirements=10,
            total_clusters=2,
            noise_count=0,
        )
        db.add(session)
        db.flush()

        for i in range(10):
            cluster_id = 0 if i < 5 else 1
            req = Requirement(
                session_id=session.id,
                req_id=f"REQ-{i+1:03d}",
                text=f"Requirement {i+1} for {'thermal' if i < 5 else 'power'} system",
                cluster_id=cluster_id,
                membership_prob=0.9,
                is_noise=False,
            )
            db.add(req)

        for cid in [0, 1]:
            cluster = Cluster(
                session_id=session.id,
                cluster_id=cid,
                label=f"Cluster {cid}",
                keywords=["keyword1", "keyword2"],
                size=5,
            )
            db.add(cluster)

        db.commit()
        return session.id
    finally:
        db.close()


class TestListSuggestionsEndpoint:
    def test_empty_list(self, client, populated_session):
        res = client.get(f"/api/suggestions?session_id={populated_session}")
        assert res.status_code == 200
        assert res.json() == []

    def test_with_persisted_suggestion(self, client, populated_session):
        # Manually insert a suggestion
        db = TestSessionLocal()
        try:
            s = RefinementSuggestion(
                session_id=populated_session,
                suggestion_type="merge",
                status="pending",
                cluster_a_id=0,
                cluster_b_id=1,
                similarity_score=0.85,
            )
            db.add(s)
            db.commit()
        finally:
            db.close()

        res = client.get(f"/api/suggestions?session_id={populated_session}")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["suggestion_type"] == "merge"
        assert data[0]["status"] == "pending"

    def test_status_filter(self, client, populated_session):
        db = TestSessionLocal()
        try:
            for status in ["pending", "applied", "rejected"]:
                s = RefinementSuggestion(
                    session_id=populated_session,
                    suggestion_type="merge",
                    status=status,
                    cluster_a_id=0,
                    cluster_b_id=1,
                )
                db.add(s)
            db.commit()
        finally:
            db.close()

        res = client.get(f"/api/suggestions?session_id={populated_session}&status=pending")
        assert res.status_code == 200
        data = res.json()
        assert all(d["status"] == "pending" for d in data)


class TestApplySuggestionEndpoint:
    def test_reject_suggestion(self, client, populated_session):
        db = TestSessionLocal()
        try:
            s = RefinementSuggestion(
                session_id=populated_session,
                suggestion_type="merge",
                status="pending",
                cluster_a_id=0,
                cluster_b_id=1,
            )
            db.add(s)
            db.commit()
            suggestion_id = s.id
        finally:
            db.close()

        res = client.post("/api/suggestions/apply", json={
            "session_id": populated_session,
            "suggestion_id": suggestion_id,
            "action": "reject",
        })
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "rejected"
        assert data["action"] == "reject"

    def test_already_applied(self, client, populated_session):
        db = TestSessionLocal()
        try:
            s = RefinementSuggestion(
                session_id=populated_session,
                suggestion_type="merge",
                status="applied",
                cluster_a_id=0,
                cluster_b_id=1,
            )
            db.add(s)
            db.commit()
            suggestion_id = s.id
        finally:
            db.close()

        res = client.post("/api/suggestions/apply", json={
            "session_id": populated_session,
            "suggestion_id": suggestion_id,
            "action": "accept",
        })
        assert res.status_code == 400

    def test_not_found(self, client, populated_session):
        res = client.post("/api/suggestions/apply", json={
            "session_id": populated_session,
            "suggestion_id": 9999,
            "action": "accept",
        })
        assert res.status_code == 404


class TestAuditLogEndpoint:
    def test_empty_audit(self, client, populated_session):
        res = client.get(f"/api/suggestions/audit?session_id={populated_session}")
        assert res.status_code == 200
        assert res.json() == []

    def test_audit_after_reject(self, client, populated_session):
        db = TestSessionLocal()
        try:
            s = RefinementSuggestion(
                session_id=populated_session,
                suggestion_type="merge",
                status="pending",
                cluster_a_id=0,
                cluster_b_id=1,
            )
            db.add(s)
            db.commit()
            suggestion_id = s.id
        finally:
            db.close()

        client.post("/api/suggestions/apply", json={
            "session_id": populated_session,
            "suggestion_id": suggestion_id,
            "action": "reject",
        })

        res = client.get(f"/api/suggestions/audit?session_id={populated_session}")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["action"] == "rejected"
        assert data[0]["suggestion_id"] == suggestion_id
