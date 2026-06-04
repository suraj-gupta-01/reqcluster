"""End-to-end tests for the DP5 dependency tree API.

Uses an isolated FastAPI app + tmp_path database per test (the same pattern as
test_feedback_api.py) so the shared global app's dependency overrides are never
mutated and tests stay order-independent.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import Base, Session, Requirement, Cluster, get_db
from api import routes


@pytest.fixture
def api_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_dependency.db'}",
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


def _seed_done_session(SessionLocal):
    db = SessionLocal()
    try:
        session = Session(
            name="Dep Session", filename="dep.csv", status="done",
            total_requirements=6, total_clusters=2, noise_count=0,
        )
        db.add(session)
        db.flush()
        rows = [
            "The thermal sensor shall generate a temperature reading.",
            "The cooling controller shall use the temperature reading as defined in REQ-001.",
            "The fan shall activate once the temperature reading exceeds the limit.",
            "The power module shall provide a regulated voltage output.",
            "The processor shall require the regulated voltage to operate.",
            "The system shall log voltage faults as specified in REQ-004.",
        ]
        clusters = [0, 0, 0, 1, 1, 1]
        for i, (text, cid) in enumerate(zip(rows, clusters)):
            db.add(Requirement(
                session_id=session.id, req_id=f"REQ-{i + 1:03d}", text=text,
                cluster_id=cid, membership_prob=0.9, is_noise=False,
            ))
        for cid, label in [(0, "Thermal Control"), (1, "Power Regulation")]:
            db.add(Cluster(
                session_id=session.id, cluster_id=cid, label=label,
                keywords=["temperature", "voltage"], size=3,
            ))
        db.commit()
        return session.id
    finally:
        db.close()


def test_generate_then_get_dependencies(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)

    gen = client.post("/api/dependencies/generate", json={"session_id": sid})
    assert gen.status_code == 200, gen.text
    body = gen.json()
    assert len(body["nodes"]) == 6
    assert "n_edges" in body["stats"]

    node_ids = {n["id"]: n["node_id"] for n in body["nodes"]}
    assert any(
        node_ids[e["source"]] == "REQ-001" and node_ids[e["target"]] == "REQ-002"
        for e in body["edges"]
    ), body["edges"]

    grouping = body["rationale"]["grouping"]
    assert len(grouping) == 2
    assert all(g["rationale"] for g in grouping)

    got = client.get(f"/api/dependencies?session_id={sid}")
    assert got.status_code == 200
    assert got.json()["stats"]["n_nodes"] == 6


def test_get_before_generate_returns_404(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)
    res = client.get(f"/api/dependencies?session_id={sid}")
    assert res.status_code == 404


def test_generate_requires_done_session(api_db):
    client, SessionLocal = api_db
    db = SessionLocal()
    try:
        s = Session(name="x", filename="x.csv", status="uploaded", total_requirements=0)
        db.add(s)
        db.commit()
        sid = s.id
    finally:
        db.close()
    res = client.post("/api/dependencies/generate", json={"session_id": sid})
    assert res.status_code == 400
