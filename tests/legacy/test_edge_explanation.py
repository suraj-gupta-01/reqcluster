"""Tests for on-demand per-edge explanation (DP5 Phase 3.2).

Covers the three properties the feature depends on: the mock provider is fully
deterministic and offline, a second request for the same edge is a cache hit
rather than a second model call, and the response parser is lenient about the
odd shapes small local models return.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
)

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from api import routes
from llm_services import edge_explanation
from llm_services.edge_explanation import (
    build_mock_explanation,
    generate_edge_explanation,
    parse_explanation_response,
)
from llm_services.prompts import build_edge_explanation_prompt, format_edge_evidence
from llm_services.providers import ProviderConfigurationError, ProviderResponseError
from models.database import (
    Base,
    Cluster,
    DependencyExplanation,
    Requirement,
    Session,
    get_db,
)


ARTIFACT_EVIDENCE = {
    "pass": "pass_0_artifact_match",
    "artifact": "temperature reading",
    "producer_artifact_name": "temperature reading",
    "consumer_artifact_name": "temperature reading",
    "synonym_match": False,
}

SEQUENTIAL_EVIDENCE = {
    "pass": "pass_2_sequential",
    "cue": "after",
    "cosine_similarity": 0.71,
    "primary_shared_token": "thermal",
    "shared_tokens": ["thermal", "limit"],
}


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def test_prompt_is_versioned_and_grounded():
    prompt = build_edge_explanation_prompt(
        "The sensor shall generate a temperature reading.",
        "The controller shall use the temperature reading.",
        "data",
        ARTIFACT_EVIDENCE,
        source_id="REQ-001",
        target_id="REQ-002",
        confidence_tier="High",
    )

    assert "reqcluster-edge-explanation-v1" in prompt
    assert "REQ-001 is a prerequisite of REQ-002" in prompt
    assert "temperature reading" in prompt
    # The anti-hallucination constraints must survive any future edit.
    assert "Do not invent" in prompt
    assert "Ground every claim in the structured evidence" in prompt


def test_evidence_formatting_bounds_and_skips_empty_values():
    rendered = format_edge_evidence(
        {"pass": "pass_2_sequential", "cosine_similarity": 0.71,
         "shared_tokens": ["a", "b"], "empty": None, "flag": True}
    )
    assert "- pass: pass_2_sequential" in rendered
    assert "- cosine_similarity: 0.71" in rendered
    assert "- shared_tokens: a, b" in rendered
    assert "- flag: true" in rendered
    assert "empty" not in rendered


def test_evidence_formatting_handles_missing_evidence():
    assert format_edge_evidence(None) == "(no structured evidence recorded)"
    assert format_edge_evidence({}) == "(no structured evidence recorded)"


# ---------------------------------------------------------------------------
# Deterministic mock provider
# ---------------------------------------------------------------------------

def test_mock_explanation_is_deterministic():
    first = build_mock_explanation("REQ-001", "REQ-002", "data", ARTIFACT_EVIDENCE, "High")
    second = build_mock_explanation("REQ-001", "REQ-002", "data", ARTIFACT_EVIDENCE, "High")
    assert first == second
    assert first


def test_mock_explanation_cites_the_shared_artifact():
    text = build_mock_explanation("REQ-001", "REQ-002", "data", ARTIFACT_EVIDENCE, "High")
    assert "temperature reading" in text
    assert "REQ-001" in text and "REQ-002" in text
    assert "high" in text.lower()


def test_mock_explanation_reports_a_synonym_match():
    evidence = {
        "pass": "pass_0_artifact_match",
        "artifact": "temp value",
        "producer_artifact_name": "temperature reading",
        "consumer_artifact_name": "temp value",
        "synonym_match": True,
        "artifact_similarity": 0.84,
    }
    text = build_mock_explanation("REQ-001", "REQ-002", "data", evidence, "High")
    assert "synonym" in text.lower()
    assert "0.84" in text


def test_mock_explanation_covers_each_pass():
    for evidence, expected in [
        (SEQUENTIAL_EVIDENCE, "after"),
        ({"pass": "pass_1_explicit_reference", "matched_text": "REQ-001"}, "REQ-001"),
        ({"pass": "pass_2_hierarchical", "cosine_similarity": 0.9}, "cluster"),
    ]:
        text = build_mock_explanation("REQ-001", "REQ-002", "data", evidence, "Medium")
        assert expected in text, (evidence, text)


def test_mock_explanation_survives_empty_evidence():
    text = build_mock_explanation("REQ-001", "REQ-002", "data", {}, "Low",
                                  rationale="Fallback rationale text.")
    assert "REQ-001" in text
    assert text


def test_generate_uses_the_mock_provider_offline():
    result = generate_edge_explanation(
        "Source text.", "Target text.", "data", ARTIFACT_EVIDENCE,
        source_id="REQ-001", target_id="REQ-002", confidence_tier="High",
        provider_name="mock",
    )
    assert result["provider"] == "mock"
    assert result["fallback"] is False
    assert "temperature reading" in result["explanation_text"]


def test_provider_failure_degrades_to_the_template(monkeypatch):
    """A configured-but-broken provider must still answer the user's click."""
    def boom(*_args, **_kwargs):
        raise ProviderConfigurationError("no endpoint configured")

    monkeypatch.setattr(edge_explanation, "generate_completion", boom)

    result = generate_edge_explanation(
        "Source text.", "Target text.", "data", ARTIFACT_EVIDENCE,
        source_id="REQ-001", target_id="REQ-002", provider_name="local",
    )

    assert result["fallback"] is True
    assert result["provider"] == "mock"
    assert "temperature reading" in result["explanation_text"]
    assert "no endpoint configured" in result["warning"]


def test_real_provider_text_is_used_when_available(monkeypatch):
    monkeypatch.setattr(
        edge_explanation, "generate_completion",
        lambda *a, **k: "REQ-001 feeds REQ-002 with the temperature reading.",
    )
    result = generate_edge_explanation(
        "Source.", "Target.", "data", ARTIFACT_EVIDENCE, provider_name="local",
    )
    assert result["provider"] == "local"
    assert result["fallback"] is False
    assert result["explanation_text"].startswith("REQ-001 feeds REQ-002")


# ---------------------------------------------------------------------------
# Response parser hardening
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw,expected_fragment",
    [
        ("A plain paragraph.", "A plain paragraph."),
        ({"explanation": "From a dict."}, "From a dict."),
        ({"text": "Under a text key."}, "Under a text key."),
        ({"content": "Under content."}, "Under content."),
        (["First element."], "First element."),
        ("```\nFenced prose.\n```", "Fenced prose."),
        ("```json\nStill prose.\n```", "Still prose."),
        ("  Padded   with\n\nwhitespace.  ", "Padded with whitespace."),
        ({"choices": "unknown shape"}, "choices"),
    ],
)
def test_parser_coerces_odd_model_shapes(raw, expected_fragment):
    assert expected_fragment in parse_explanation_response(raw)


@pytest.mark.parametrize("raw", [None, "", "   ", [], {"explanation": ""}])
def test_parser_rejects_empty_responses(raw):
    with pytest.raises(ProviderResponseError):
        parse_explanation_response(raw)


def test_parser_bounds_very_long_responses():
    result = parse_explanation_response("word " * 5000)
    assert len(result) <= edge_explanation.MAX_EXPLANATION_CHARS


# ---------------------------------------------------------------------------
# API + caching
# ---------------------------------------------------------------------------

@pytest.fixture
def api_db(tmp_path):
    engine = create_engine(
        f"sqlite:///{tmp_path / 'test_explain.db'}",
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
            name="Explain Session", filename="explain.csv", status="done",
            total_requirements=4, total_clusters=1, noise_count=0,
        )
        db.add(session)
        db.flush()
        rows = [
            "The thermal sensor shall generate a temperature reading.",
            "The cooling controller shall use the temperature reading as defined in REQ-001.",
            "The fan shall activate once the temperature reading exceeds the limit.",
            "The logger shall record the temperature reading for later analysis.",
        ]
        for i, text in enumerate(rows):
            db.add(Requirement(
                session_id=session.id, req_id=f"REQ-{i + 1:03d}", text=text,
                cluster_id=0, membership_prob=0.9, is_noise=False,
            ))
        db.add(Cluster(
            session_id=session.id, cluster_id=0, label="Thermal Control",
            keywords=["temperature"], size=4,
        ))
        db.commit()
        return session.id
    finally:
        db.close()


def _first_edge(client, sid):
    body = client.post("/api/dependencies/generate", json={"session_id": sid}).json()
    assert body["edges"], "fixture produced no edges to explain"
    return body["edges"][0]


def test_generated_edges_carry_evidence_and_a_confidence_tier(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)
    body = client.post("/api/dependencies/generate", json={"session_id": sid}).json()

    for edge in body["edges"]:
        assert edge["confidence_tier"] in ("High", "Medium", "Low")
        assert isinstance(edge["evidence"], dict)
        assert edge["evidence"].get("pass"), edge
    assert body["stats"]["confidence_tier_counts"]


def test_explain_endpoint_returns_evidence_and_prose(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)
    edge = _first_edge(client, sid)

    res = client.post(
        f"/api/dependencies/{sid}/explain",
        json={"source_index": edge["source"], "target_index": edge["target"]},
    )
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["explanation_text"]
    assert body["cached"] is False
    assert body["provider"] == "mock"
    assert body["relation"] == edge["relation"]
    assert body["confidence_tier"] == edge["confidence_tier"]
    assert body["evidence"] == edge["evidence"]


def test_second_request_is_a_cache_hit(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)
    edge = _first_edge(client, sid)
    payload = {"source_index": edge["source"], "target_index": edge["target"]}

    first = client.post(f"/api/dependencies/{sid}/explain", json=payload).json()
    second = client.post(f"/api/dependencies/{sid}/explain", json=payload).json()

    assert first["cached"] is False
    assert second["cached"] is True
    assert second["explanation_text"] == first["explanation_text"]

    db = SessionLocal()
    try:
        assert db.query(DependencyExplanation).filter(
            DependencyExplanation.session_id == sid
        ).count() == 1
    finally:
        db.close()


def test_force_refresh_bypasses_the_cache(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)
    edge = _first_edge(client, sid)
    payload = {"source_index": edge["source"], "target_index": edge["target"]}

    client.post(f"/api/dependencies/{sid}/explain", json=payload)
    refreshed = client.post(
        f"/api/dependencies/{sid}/explain", json={**payload, "force_refresh": True}
    ).json()

    assert refreshed["cached"] is False

    db = SessionLocal()
    try:
        assert db.query(DependencyExplanation).filter(
            DependencyExplanation.session_id == sid
        ).count() == 1
    finally:
        db.close()


def test_explain_unknown_edge_returns_404(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)
    _first_edge(client, sid)

    res = client.post(
        f"/api/dependencies/{sid}/explain",
        json={"source_index": 0, "target_index": 999},
    )
    assert res.status_code == 404


def test_explain_before_generate_returns_404(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)

    res = client.post(
        f"/api/dependencies/{sid}/explain",
        json={"source_index": 0, "target_index": 1},
    )
    assert res.status_code == 404


def test_explain_rejects_negative_indices(api_db):
    client, SessionLocal = api_db
    sid = _seed_done_session(SessionLocal)

    res = client.post(
        f"/api/dependencies/{sid}/explain",
        json={"source_index": -1, "target_index": 1},
    )
    assert res.status_code == 422
