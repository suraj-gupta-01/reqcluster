import json

import numpy as np
import pytest

pytest.importorskip("sqlalchemy")

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker

from api import routes
from models.database import Base, EnrichedRequirement, Requirement, Session as DbSessionModel, get_db
from services.enrichment_service import get_enriched_texts_for_session_ordered


REQUIREMENT_TEXTS = [
    "The controller shall store audit logs.",
    "The controller shall retrieve audit logs.",
    "The controller shall archive audit logs.",
    "The controller shall delete audit logs.",
]


@pytest.fixture
def api_db(tmp_path, monkeypatch):
    monkeypatch.setenv("REQCLUSTER_LLM_PROVIDER", "mock")
    engine = create_engine(
        f"sqlite:///{tmp_path / 'reqcluster_test.db'}",
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


def seed_session(SessionLocal, texts=REQUIREMENT_TEXTS):
    db = SessionLocal()
    try:
        session = DbSessionModel(
            name="test",
            filename="test.csv",
            status="uploaded",
            total_requirements=len(texts),
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        for idx, text in enumerate(texts):
            db.add(
                Requirement(
                    session_id=session.id,
                    req_id=f"REQ-{idx + 1}",
                    text=text,
                    module="",
                    section="",
                )
            )
        db.commit()
        return session.id
    finally:
        db.close()


def patch_pipeline(monkeypatch):
    calls = {"count": 0}

    def fake_run_pipeline(**kwargs):
        calls["count"] += 1
        calls["kwargs"] = kwargs
        n = len(kwargs["texts"])
        labels = np.array([0 if idx < n // 2 else 1 for idx in range(n)], dtype=int)
        probabilities = np.ones(n, dtype=np.float32)
        embeddings = np.eye(n, 384, dtype=np.float32)
        return {
            "embeddings": embeddings,
            "embeddings_10d": embeddings[:, :10],
            "embeddings_2d": embeddings[:, :2],
            "labels": labels,
            "probabilities": probabilities,
            "cluster_info": {
                0: {"label": "First", "keywords": ["first"], "size": int(np.sum(labels == 0))},
                1: {"label": "Second", "keywords": ["second"], "size": int(np.sum(labels == 1))},
            },
            "graph_data": {"nodes": [{"id": idx} for idx in range(n)], "edges": []},
            "n_clusters": 2,
            "noise_count": 0,
            "embedding_mode": kwargs["embedding_mode"],
            "embedding_comparison": {"ok": True} if kwargs.get("enable_embedding_comparison") else None,
            "ablation_report": {"ok": True} if kwargs.get("run_ablation") else None,
        }

    monkeypatch.setattr(routes, "run_pipeline", fake_run_pipeline)
    return calls


def test_enriched_requirement_table_is_created(api_db):
    _client, _SessionLocal, engine = api_db

    tables = inspect(engine).get_table_names()

    assert "enriched_requirements" in tables


def test_post_enrich_persists_rows_and_json_fields(api_db):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)

    response = client.post("/api/enrich", json={"session_id": session_id})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "complete"
    assert body["succeeded"] == 4
    assert body["provider"] == "mock"
    assert "sk-" not in json.dumps(body)

    db = SessionLocal()
    try:
        rows = (
            db.query(EnrichedRequirement)
            .filter(EnrichedRequirement.session_id == session_id)
            .order_by(EnrichedRequirement.requirement_db_id.asc())
            .all()
        )
        assert len(rows) == 4
        assert rows[0].requirement_db_id is not None
        assert rows[0].requirement_id == "REQ-1"
        assert isinstance(rows[0].domain_terms_json, list)
        assert isinstance(rows[0].quality_report_json, dict)
    finally:
        db.close()


def test_duplicate_enrichment_reuses_rows_and_force_refresh_updates(api_db):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)

    first = client.post("/api/enrich", json={"session_id": session_id})
    second = client.post("/api/enrich", json={"session_id": session_id})
    refreshed = client.post(
        "/api/enrich",
        json={"session_id": session_id, "force_refresh": True, "use_cache": False},
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert refreshed.status_code == 200

    db = SessionLocal()
    try:
        assert db.query(EnrichedRequirement).filter_by(session_id=session_id).count() == 4
    finally:
        db.close()


def test_enrichment_status_and_results_are_ordered(api_db):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)
    client.post("/api/enrich", json={"session_id": session_id})

    status = client.get(f"/api/enrich/status/{session_id}")
    results = client.get(f"/api/enrich/results?session_id={session_id}")

    assert status.status_code == 200
    assert status.json()["succeeded"] == 4
    assert status.json()["pending"] == 0
    assert results.status_code == 200
    assert [row["requirement_id"] for row in results.json()] == [
        "REQ-1",
        "REQ-2",
        "REQ-3",
        "REQ-4",
    ]


def test_post_enrich_invalid_session_and_provider_are_safe(api_db):
    client, _SessionLocal, _engine = api_db

    missing = client.post("/api/enrich", json={"session_id": 999})
    invalid_provider = client.post(
        "/api/enrich",
        json={"session_id": 1, "provider_name": "not_allowed"},
    )
    invalid_batch = client.post(
        "/api/enrich",
        json={"session_id": 1, "batch_size": 65},
    )
    invalid_timeout = client.post(
        "/api/enrich",
        json={"session_id": 1, "timeout_seconds": 999},
    )

    assert missing.status_code == 404
    assert "Session 999 not found" in missing.json()["detail"]
    assert invalid_provider.status_code == 422
    assert invalid_batch.status_code == 422
    assert invalid_timeout.status_code == 422


def test_openai_provider_error_does_not_expose_api_key(api_db, monkeypatch):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)
    monkeypatch.setenv("REQCLUSTER_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("REQCLUSTER_LLM_API_KEY", "sk-secret-test-key")
    monkeypatch.delenv("REQCLUSTER_LLM_MODEL", raising=False)

    response = client.post(
        "/api/enrich",
        json={"session_id": session_id, "provider_name": "openai_compatible"},
    )

    assert response.status_code == 400
    assert "sk-secret-test-key" not in response.text
    assert "sk-" not in response.text


def test_cluster_base_mode_still_works_without_enrichment(api_db, monkeypatch):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)
    calls = patch_pipeline(monkeypatch)

    response = client.post(
        "/api/cluster",
        json={"session_id": session_id, "embedding_mode": "base"},
    )

    assert response.status_code == 200
    assert response.json()["total_clusters"] == 2
    assert calls["kwargs"]["embedding_mode"] == "base"
    assert calls["kwargs"]["enriched_texts"] is None


def test_cluster_hybrid_requires_enrichment(api_db, monkeypatch):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)
    patch_pipeline(monkeypatch)

    response = client.post(
        "/api/cluster",
        json={"session_id": session_id, "embedding_mode": "hybrid"},
    )

    assert response.status_code == 400
    assert "Run /api/enrich" in response.json()["detail"]


def test_cluster_hybrid_uses_persisted_enriched_texts(api_db, monkeypatch):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)
    client.post("/api/enrich", json={"session_id": session_id})
    calls = patch_pipeline(monkeypatch)

    response = client.post(
        "/api/cluster",
        json={
            "session_id": session_id,
            "embedding_mode": "hybrid",
            "enable_embedding_comparison": True,
            "run_ablation": True,
        },
    )

    assert response.status_code == 200
    assert calls["kwargs"]["embedding_mode"] == "hybrid"
    assert len(calls["kwargs"]["enriched_texts"]) == 4
    assert calls["kwargs"]["enriched_texts"][0].startswith("Original requirement:")
    assert calls["kwargs"]["enable_embedding_comparison"] is True
    assert calls["kwargs"]["run_ablation"] is True


def test_alignment_mismatch_is_detected_for_cluster(api_db, monkeypatch):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)
    client.post("/api/enrich", json={"session_id": session_id})
    patch_pipeline(monkeypatch)

    db = SessionLocal()
    try:
        row = db.query(EnrichedRequirement).filter_by(session_id=session_id).first()
        row.requirement_text_hash = "0" * 64
        db.commit()
    finally:
        db.close()

    response = client.post(
        "/api/cluster",
        json={"session_id": session_id, "embedding_mode": "hybrid"},
    )

    assert response.status_code == 400
    assert "Run /api/enrich" in response.json()["detail"]


def test_malformed_persisted_json_is_handled_safely(api_db):
    client, SessionLocal, _engine = api_db
    session_id = seed_session(SessionLocal)
    client.post("/api/enrich", json={"session_id": session_id})

    db = SessionLocal()
    try:
        row = db.query(EnrichedRequirement).filter_by(session_id=session_id).first()
        row.domain_terms_json = "{bad json"
        db.commit()
    finally:
        db.close()

    results = client.get(f"/api/enrich/results?session_id={session_id}")

    assert results.status_code == 200
    assert results.json()[0]["domain_terms"] == []

    db = SessionLocal()
    try:
        with pytest.raises(Exception):
            get_enriched_texts_for_session_ordered(db, session_id)
    finally:
        db.close()
