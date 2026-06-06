import os
import sys
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from models.database import Base, Session, Requirement, get_db
from api import routes

@pytest.fixture
def api_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_paginated_req.db'}",
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

    # Override SessionLocal inside routes to use the test database
    original_session_local = routes.SessionLocal
    routes.SessionLocal = TestingSessionLocal

    with TestClient(app) as client:
        yield client, TestingSessionLocal

    routes.SessionLocal = original_session_local
    engine.dispose()

def _seed(SessionLocal):
    db = SessionLocal()
    try:
        s = Session(name="test_session", filename="test.csv", status="uploaded",
                    total_requirements=5, total_clusters=0)
        db.add(s)
        db.flush()
        
        # Add 5 distinct requirements for test coverage
        reqs = [
            Requirement(session_id=s.id, req_id="REQ-A", text="High performance server code", module="Core", section="Backend", membership_prob=0.9, is_noise=False),
            Requirement(session_id=s.id, req_id="REQ-B", text="Premium visual user interface design", module="UI", section="Frontend", membership_prob=0.85, is_noise=False),
            Requirement(session_id=s.id, req_id="REQ-C", text="Database index mapping", module="Core", section="DB", membership_prob=0.5, is_noise=True),
            Requirement(session_id=s.id, req_id="REQ-D", text="Client side data binding", module="UI", section="Frontend", membership_prob=0.7, is_noise=False),
            Requirement(session_id=s.id, req_id="REQ-E", text="API rate limiting", module="Core", section="API", membership_prob=0.95, is_noise=False),
        ]
        for r in reqs:
            db.add(r)
        db.commit()
        return s.id
    finally:
        db.close()

def test_requirements_no_pagination(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    res = client.get(f"/api/requirements?session_id={sid}")
    assert res.status_code == 200
    assert len(res.json()) == 5
    assert res.headers.get("X-Total-Count") == "5"

def test_requirements_pagination_slices(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    # First page
    res = client.get(f"/api/requirements?session_id={sid}&page=1&page_size=2")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 2
    assert res.headers.get("X-Total-Count") == "5"
    assert items[0]["req_id"] == "REQ-A"
    assert items[1]["req_id"] == "REQ-B"

    # Second page
    res = client.get(f"/api/requirements?session_id={sid}&page=2&page_size=2")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 2
    assert items[0]["req_id"] == "REQ-C"
    assert items[1]["req_id"] == "REQ-D"

def test_requirements_search_filter(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    # Search for "core" in module
    res = client.get(f"/api/requirements?session_id={sid}&search=core")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 3
    assert res.headers.get("X-Total-Count") == "3"
    req_ids = {item["req_id"] for item in items}
    assert req_ids == {"REQ-A", "REQ-C", "REQ-E"}

    # Search for "frontend" in section
    res = client.get(f"/api/requirements?session_id={sid}&search=frontend")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 2
    assert {item["req_id"] for item in items} == {"REQ-B", "REQ-D"}

def test_requirements_is_noise_filter(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    # Filter for noise only
    res = client.get(f"/api/requirements?session_id={sid}&is_noise=true")
    assert res.status_code == 200
    items = res.json()
    assert len(items) == 1
    assert items[0]["req_id"] == "REQ-C"

    # Filter for clustered only
    res = client.get(f"/api/requirements?session_id={sid}&is_noise=false")
    assert res.status_code == 200
    assert len(res.json()) == 4

def test_requirements_sorting(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    # Sort by membership_prob descending
    res = client.get(f"/api/requirements?session_id={sid}&sort_field=membership_prob&sort_dir=desc")
    assert res.status_code == 200
    items = res.json()
    assert items[0]["req_id"] == "REQ-E" # 0.95
    assert items[1]["req_id"] == "REQ-A" # 0.90
    assert items[2]["req_id"] == "REQ-B" # 0.85
    assert items[3]["req_id"] == "REQ-D" # 0.70
    assert items[4]["req_id"] == "REQ-C" # 0.50

    # Sort by req_id descending
    res = client.get(f"/api/requirements?session_id={sid}&sort_field=req_id&sort_dir=desc")
    assert res.status_code == 200
    items = res.json()
    assert items[0]["req_id"] == "REQ-E"
    assert items[4]["req_id"] == "REQ-A"
