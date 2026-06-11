import os
import sys
import time
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from models.database import (
    Base, Session, Requirement, get_db,
)
from api import routes


@pytest.fixture
def api_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_async_cluster.db'}",
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
                    total_requirements=10, total_clusters=0)
        db.add(s)
        db.flush()
        
        # Add 10 requirements
        for i in range(10):
            r = Requirement(
                session_id=s.id,
                req_id=f"REQ-{i+1:03d}",
                text=f"Sample engineering requirement text for requirement number {i+1}.",
                is_noise=False,
            )
            db.add(r)
        db.commit()
        return s.id
    finally:
        db.close()


def test_requirements_pagination(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    # 1. Test full requirements list
    res = client.get(f"/api/requirements?session_id={sid}")
    assert res.status_code == 200
    assert len(res.json()) == 10

    # 2. Test paginated first slice (page=1, page_size=4)
    res_page1 = client.get(f"/api/requirements?session_id={sid}&page=1&page_size=4")
    assert res_page1.status_code == 200
    items = res_page1.json()
    assert len(items) == 4
    assert items[0]["req_id"] == "REQ-001"
    assert items[3]["req_id"] == "REQ-004"

    # 3. Test paginated second slice (page=2, page_size=4)
    res_page2 = client.get(f"/api/requirements?session_id={sid}&page=2&page_size=4")
    assert res_page2.status_code == 200
    items2 = res_page2.json()
    assert len(items2) == 4
    assert items2[0]["req_id"] == "REQ-005"
    assert items2[3]["req_id"] == "REQ-008"

    # 4. Test remaining paginated slice (page=3, page_size=4)
    res_page3 = client.get(f"/api/requirements?session_id={sid}&page=3&page_size=4")
    assert res_page3.status_code == 200
    items3 = res_page3.json()
    assert len(items3) == 2
    assert items3[0]["req_id"] == "REQ-009"
    assert items3[1]["req_id"] == "REQ-010"


def test_async_clustering_workflow(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    # Trigger async clustering (should return 202 immediately)
    res = client.post("/api/cluster", json={
        "session_id": sid,
        "min_cluster_size": 2,
        "min_samples": 1,
        "similarity_threshold": 0.5,
        "embedding_mode": "base",
    })
    assert res.status_code == 202
    body = res.json()
    assert body["session_id"] == sid
    assert body["status"] == "processing"

    # Test the 1-second TTL cache on progress polling
    t1 = time.time()
    prog1 = client.get(f"/api/progress/{sid}").json()
    
    # Instant follow-up call should hit cache (and thus be identical / same ref or content)
    prog2 = client.get(f"/api/progress/{sid}").json()
    assert prog1 == prog2

    # Wait for the background task thread to finish (max 10 seconds since it's running mocks)
    max_wait = 15.0
    elapsed = 0.0
    status = "processing"
    
    while elapsed < max_wait:
        # Clear progress cache by waiting 1.1 seconds or manually check DB status
        time.sleep(0.5)
        elapsed += 0.5
        
        # Check DB status directly to bypass progress dict cache if needed
        db = SessionLocal()
        s = db.query(Session).filter(Session.id == sid).first()
        status = s.status
        db.close()
        
        if status in ("done", "error"):
            break

    assert status == "done"

    # Verify progress polling gets done
    # Wait 1.1s to guarantee TTL expiration of /progress cache
    time.sleep(1.1)
    prog_final = client.get(f"/api/progress/{sid}").json()
    assert prog_final["step"] == "done"
    assert prog_final["progress"] == 100
