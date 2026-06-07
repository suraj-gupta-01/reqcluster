"""End-to-end tests for the Phase 5 export endpoint."""

import os
import sys
import xml.etree.ElementTree as ET

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
        f"sqlite:///{tmp_path / 'test_export.db'}",
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


def _seed(SessionLocal):
    db = SessionLocal()
    try:
        s = Session(name="Exp", filename="exp.csv", status="done",
                    total_requirements=2, total_clusters=1)
        db.add(s)
        db.flush()
        for i in range(2):
            db.add(Requirement(
                session_id=s.id, req_id=f"REQ-{i + 1:03d}",
                text=f"The system shall do thing {i}.", cluster_id=0,
                module="M", section="S", membership_prob=0.9, is_noise=False,
            ))
        db.add(Cluster(session_id=s.id, cluster_id=0, label="Group", keywords=["x"], size=2))
        db.commit()
        return s.id
    finally:
        db.close()


@pytest.mark.parametrize("fmt,ctype", [
    ("reqif", "application/xml"),
    ("sysml", "application/xml"),
    ("jama", "application/json"),
    ("csv", "text/csv"),
    ("pdf", "application/pdf"),
])
def test_export_formats(api_db, fmt, ctype):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)
    res = client.get(f"/api/export/{fmt}?session_id={sid}")
    assert res.status_code == 200, res.text
    assert ctype in res.headers["content-type"]
    assert "attachment" in res.headers.get("content-disposition", "")
    if fmt in ("reqif", "sysml"):
        ET.fromstring(res.text)  # parseable XML


def test_export_unsupported_format(api_db):
    client, SessionLocal = api_db
    sid = _seed(SessionLocal)
    res = client.get(f"/api/export/docx?session_id={sid}")
    assert res.status_code == 400


def test_export_missing_session(api_db):
    client, _ = api_db
    res = client.get("/api/export/csv?session_id=999")
    assert res.status_code == 404


def test_concurrent_reads_do_not_error(api_db):
    """Many parallel reads must not raise (regression for the SQLite cursor /
    StaticPool concurrency bug)."""
    import concurrent.futures as cf

    client, SessionLocal = api_db
    sid = _seed(SessionLocal)

    def hit(_):
        return client.get(f"/api/clusters?session_id={sid}").status_code

    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        codes = list(ex.map(hit, range(60)))
    assert all(c == 200 for c in codes), codes


def test_csv_export_sanitizes_formula_injection(api_db):
    """A requirement whose text starts with '=' must not export as a live formula."""
    client, SessionLocal = api_db
    db = SessionLocal()
    try:
        s = Session(name="inj", filename="inj.csv", status="done",
                    total_requirements=1, total_clusters=1)
        db.add(s); db.flush()
        db.add(Requirement(session_id=s.id, req_id="R1", text="=1+2",
                           cluster_id=0, module="M", section="S", is_noise=False))
        db.add(Cluster(session_id=s.id, cluster_id=0, label="G", keywords=["x"], size=1))
        db.commit()
        sid = s.id
    finally:
        db.close()
    res = client.get(f"/api/export/csv?session_id={sid}")
    assert res.status_code == 200
    assert "'=1+2" in res.text          # neutralized
    assert ",=1+2" not in res.text       # not a raw formula cell
