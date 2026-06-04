"""End-to-end tests for Phase 5 active-learning API (isolated app per test)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import (
    Base, Session, Requirement, Cluster, FeedbackCorrection, ConstraintPair, get_db,
)
from api import routes


@pytest.fixture
def api_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_al.db'}",
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
    with TestClient(app) as client:
        yield client, TestingSessionLocal
    engine.dispose()


def _seed(SessionLocal, with_constraint=True):
    db = SessionLocal()
    try:
        s = Session(name="al", filename="al.csv", status="done",
                    total_requirements=6, total_clusters=2)
        db.add(s)
        db.flush()
        reqs = []
        for i in range(6):
            cid = 0 if i < 3 else 1
            r = Requirement(
                session_id=s.id, req_id=f"REQ-{i + 1:03d}",
                text=f"requirement {i} about {'thermal' if cid == 0 else 'power'}",
                cluster_id=cid, membership_prob=0.5 if i == 1 else 0.95, is_noise=False,
            )
            db.add(r)
            reqs.append(r)
        for cid in (0, 1):
            db.add(Cluster(session_id=s.id, cluster_id=cid, label=f"C{cid}",
                           keywords=["k"], size=3))
        db.flush()
        if with_constraint:
            fc = FeedbackCorrection(
                session_id=s.id, requirement_id=reqs[0].id, previous_cluster_id=0,
                new_cluster_id=1, confidence_score=1.0, status="pending",
            )
            db.add(fc)
            db.flush()
            # must-link req0 with req3 (both should be together)
            db.add(ConstraintPair(
                session_id=s.id, requirement_a_id=reqs[0].id, requirement_b_id=reqs[3].id,
                constraint_type="must-link", feedback_id=fc.id,
            ))
        db.commit()
        return s.id
    finally:
        db.close()


def test_constrained_recluster_records_iteration(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal, with_constraint=True)

    res = client.post("/api/cluster/constrained", json={"session_id": sid})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["iteration"] == 1
    assert body["constraints"]["must_link_pairs"] == 1
    assert "quality" in body

    hist = client.get(f"/api/quality/history?session_id={sid}")
    assert hist.status_code == 200
    assert hist.json()["count"] == 1


def test_constrained_recluster_requires_constraints(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal, with_constraint=False)
    res = client.post("/api/cluster/constrained", json={"session_id": sid})
    assert res.status_code == 400


def test_uncertainty_queue(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal, with_constraint=False)
    res = client.get(f"/api/active-learning/queue?session_id={sid}&top_k=3")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 3
    # The 0.5-membership requirement should be among the most uncertain.
    assert any(item["req_id"] == "REQ-002" for item in body["queue"])
    assert all("requirement_id" in item for item in body["queue"])
