import json

import pytest

from reqcluster.config import get_settings
from reqcluster.llm.enrich import (
    ExpansionOut,
    build_quality_report,
    evaluate_expansion_quality,
    expand_all,
    extract_domain_vocabulary,
    mock_expansion,
    parse_expansion,
)
from reqcluster.llm.gateway import LLMGateway


def valid(**o):
    d = {"expanded_text": "Original requirement: The system shall store audit records.",
         "domain_terms": ["audit records", "store"], "functional_intent": "store audit records",
         "mentioned_components": [], "assumptions": [], "confidence": 0.8, "warnings": []}
    d.update(o)
    return d


def test_parse_valid():
    p = parse_expansion(valid())
    assert p.domain_terms == ["audit records", "store"] and p.confidence == 0.8


def test_parse_rejects_non_object_and_missing():
    with pytest.raises(ValueError):
        parse_expansion([])
    with pytest.raises(ValueError):
        parse_expansion({"expanded_text": "x"})


def test_parse_invalid_confidence_and_control_chars():
    p = parse_expansion(valid(expanded_text="Original\u0000 requirement: keep audit.",
                              confidence="high"))
    assert "\u0000" not in p.expanded_text and p.confidence == 0.0
    assert any("confidence" in w.lower() for w in p.warnings)


def test_parse_caps_and_clamps():
    p = parse_expansion(valid(expanded_text="x" * 20_000, confidence=7,
                              assumptions=[f"a{i}" for i in range(40)]))
    assert len(p.expanded_text) == 12_000 and p.confidence == 1.0 and len(p.assumptions) == 20


def test_mock_expansion_deterministic_and_keeps_numbers():
    t = "The controller shall retain 5 alarm events."
    a, b = mock_expansion(t), mock_expansion(t)
    assert a == b and t in a.expanded_text and "10" not in a.expanded_text
    json.dumps(a.model_dump())
    with pytest.raises(ValueError):
        mock_expansion("  ")


def test_vocabulary_stable_stopwords_removed():
    texts = ["The system shall support audit log storage and audit log retrieval.",
             "Audit log storage shall include encrypted storage events."]
    first = extract_domain_vocabulary(texts, top_n=20)
    assert first == extract_domain_vocabulary(texts, top_n=20)
    assert "shall" not in first and "system" not in first
    assert {"audit", "audit log", "audit log storage"} <= set(first)
    assert extract_domain_vocabulary([], top_n=10) == []


def test_quality_warnings():
    def w(o, e):
        return " ".join(evaluate_expansion_quality(o, e, [], 0.9)["warnings"])

    assert "numeric" in w("The system may store audit events.", "The system shall store 10 audit events.")
    assert "strengthens" in w("The system may store audit events.", "The system shall store audit events.")
    assert "overlap" in w("The system shall store audit events.", "Unrelated payment invoice processing.")
    assert "too short" in w("The system shall store audit events for operator review.", "store")
    assert "too long" in w("The system shall store audit events.",
                           "The system shall store audit events. " * 10)


def test_expand_all_mock_fallback_and_report():
    gw = LLMGateway(get_settings())
    reqs = [(1, "The controller shall retain 5 alarm events."), (2, ""), (3, "The API shall log errors.")]
    out = expand_all(reqs, gw)
    assert [e.requirement_id for e in out] == [1, 2, 3]
    assert [e.status for e in out] == ["succeeded", "failed", "succeeded"]
    assert out[0].source == "fallback" and out[0].expanded_text
    rep = build_quality_report(reqs, out)
    assert (rep["total"], rep["succeeded"], rep["failed"]) == (3, 2, 1)


def test_expansion_out_requires_fields():
    with pytest.raises(ValueError):
        ExpansionOut.model_validate({"expanded_text": "x"})


def test_expand_all_real_provider_fallback_is_failed(monkeypatch):
    from dataclasses import replace

    from reqcluster.llm import gateway as gw
    monkeypatch.setattr(gw, "_sleep", lambda s: None)
    s = replace(get_settings(), llm_provider="openai", llm_base_url="", llm_max_retries=0)
    out = expand_all([(1, "The system shall store audit records.")], LLMGateway(s))
    assert out[0].status == "failed" and out[0].source == "fallback" and out[0].error
