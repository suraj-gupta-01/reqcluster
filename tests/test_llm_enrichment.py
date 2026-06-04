import asyncio
import json

import numpy as np
import pytest

from core import pipeline
from llm_services import enrichment
from llm_services.enrichment import (
    enrich_requirements,
    enrich_then_prepare_pipeline_inputs,
    extract_enriched_texts,
)
from llm_services.prompts import PROMPT_VERSION, build_requirement_expansion_prompt
from llm_services.providers import (
    LocalLLMProvider,
    MockLLMProvider,
    OpenAICompatibleProvider,
    ParsedExpansion,
    parse_expansion_response,
)
from llm_services.quality import evaluate_expansion_quality
from llm_services.vocabulary import extract_domain_vocabulary


def valid_response(**overrides):
    payload = {
        "expanded_text": "Original requirement: The system shall store audit records.",
        "domain_terms": ["audit records", "store"],
        "functional_intent": "store audit records",
        "mentioned_components": [],
        "assumptions": [],
        "confidence": 0.8,
        "warnings": [],
    }
    payload.update(overrides)
    return json.dumps(payload)


def run(coro):
    return asyncio.run(coro)


def test_prompt_builder_includes_requirement_and_strict_json_rules():
    prompt = build_requirement_expansion_prompt(
        "The telemetry service shall store status events.",
        ["telemetry service"],
    )

    assert "The telemetry service shall store status events." in prompt
    assert "Return strict JSON only" in prompt
    assert "Do not add new obligations" in prompt
    assert "Do not invent numeric thresholds" in prompt
    assert PROMPT_VERSION == "reqcluster-requirement-expansion-v1"


def test_parser_accepts_valid_json():
    parsed = parse_expansion_response(valid_response())

    assert parsed.expanded_text.startswith("Original requirement")
    assert parsed.domain_terms == ["audit records", "store"]
    assert parsed.confidence == 0.8


def test_parser_rejects_markdown_fences_and_arrays_and_huge_responses():
    with pytest.raises(Exception):
        parse_expansion_response("```json\n{}\n```")

    with pytest.raises(Exception):
        parse_expansion_response("[]")

    with pytest.raises(Exception):
        parse_expansion_response("x" * 65_000)


def test_parser_handles_invalid_confidence_and_control_characters():
    parsed = parse_expansion_response(
        valid_response(
            expanded_text="Original\u0000 requirement: keep audit records.",
            confidence="high",
        )
    )

    assert "\u0000" not in parsed.expanded_text
    assert parsed.confidence == 0.0
    assert any("confidence" in warning.lower() for warning in parsed.warnings)


def test_mock_provider_is_deterministic_offline_and_preserves_numbers(monkeypatch):
    def fail_network(*args, **kwargs):
        raise AssertionError("network should not be used by mock provider")

    monkeypatch.setattr("llm_services.providers.urllib.request.urlopen", fail_network)
    provider = MockLLMProvider()
    text = "The controller shall retain 5 alarm events."

    first = run(provider.expand_requirement(text))
    second = run(provider.expand_requirement(text))

    assert first == second
    json.dumps(first.to_dict())
    assert "5" in first.expanded_text
    assert "10" not in first.expanded_text
    assert text in first.expanded_text


def test_local_provider_rejects_unsafe_url_scheme():
    with pytest.raises(Exception):
        LocalLLMProvider("file:///tmp/model")


def test_openai_provider_missing_key_fails_without_secret_exposure(monkeypatch):
    monkeypatch.setenv("REQCLUSTER_LLM_BASE_URL", "https://example.invalid/v1")
    monkeypatch.delenv("REQCLUSTER_LLM_API_KEY", raising=False)
    monkeypatch.setenv("REQCLUSTER_LLM_MODEL", "test-model")

    with pytest.raises(Exception) as exc_info:
        OpenAICompatibleProvider.from_env()

    assert "sk-" not in str(exc_info.value)


class CountingProvider:
    name = "counting"
    model = "counting-v1"

    def __init__(self, fail_on=None):
        self.calls = 0
        self.fail_on = fail_on or set()

    async def expand_requirement(self, requirement_text, **kwargs):
        self.calls += 1
        if requirement_text in self.fail_on:
            raise RuntimeError("planned provider failure")
        return ParsedExpansion(
            expanded_text=f"Original requirement: {requirement_text} Domain vocabulary from the requirement: storage.",
            domain_terms=["storage"],
            functional_intent=requirement_text,
            mentioned_components=[],
            assumptions=[],
            confidence=0.75,
            warnings=[],
        )


def test_enrichment_preserves_order_and_is_json_serializable(monkeypatch):
    provider = CountingProvider()
    monkeypatch.setattr(enrichment, "get_provider", lambda name: provider)

    batch = run(
        enrich_requirements(
            ["first storage requirement", "second storage requirement"],
            ["REQ-B", "REQ-A"],
            provider_name="counting",
            use_cache=False,
        )
    )

    assert [result.requirement_id for result in batch.results] == ["REQ-B", "REQ-A"]
    assert extract_enriched_texts(batch)[0].startswith("Original requirement: first")
    json.dumps(batch.to_dict())


def test_enrichment_partial_failure_and_extract_alignment(monkeypatch):
    provider = CountingProvider(fail_on={"bad requirement"})
    monkeypatch.setattr(enrichment, "get_provider", lambda name: provider)

    batch = run(
        enrich_requirements(
            ["good requirement", "bad requirement", "last requirement"],
            ["REQ-1", "REQ-2", "REQ-3"],
            provider_name="counting",
            use_cache=False,
        )
    )

    assert batch.succeeded == 2
    assert batch.failed == 1
    assert batch.errors[0]["index"] == 1
    assert extract_enriched_texts(batch)[1] is None
    assert extract_enriched_texts(batch)[2].startswith("Original requirement: last")


def test_enrichment_fail_fast_raises(monkeypatch):
    provider = CountingProvider(fail_on={"bad requirement"})
    monkeypatch.setattr(enrichment, "get_provider", lambda name: provider)

    with pytest.raises(RuntimeError):
        run(
            enrich_requirements(
                ["bad requirement"],
                provider_name="counting",
                fail_fast=True,
                use_cache=False,
            )
        )


def test_enrichment_cache_hit_miss_and_corruption(tmp_path, monkeypatch):
    monkeypatch.setattr(enrichment, "CACHE_DIR", tmp_path)
    provider = CountingProvider()
    monkeypatch.setattr(enrichment, "get_provider", lambda name: provider)

    first = run(enrich_requirements(["cacheable storage requirement"], provider_name="counting"))
    second = run(enrich_requirements(["cacheable storage requirement"], provider_name="counting"))

    assert first.succeeded == 1
    assert second.succeeded == 1
    assert provider.calls == 1

    cache_file = next(tmp_path.glob("llm_enrichment_*.json"))
    cache_file.write_text("{bad json", encoding="utf-8")

    third = run(enrich_requirements(["cacheable storage requirement"], provider_name="counting"))

    assert third.succeeded == 1
    assert provider.calls == 2


def test_vocabulary_extracts_ngrams_removes_stopwords_and_is_stable():
    texts = [
        "The system shall support audit log storage and audit log retrieval.",
        "Audit log storage shall include encrypted storage events.",
    ]

    first = extract_domain_vocabulary(texts, top_n=20)
    second = extract_domain_vocabulary(texts, top_n=20)

    assert first == second
    assert "shall" not in first
    assert "system" not in first
    assert "audit" in first
    assert "audit log" in first
    assert "audit log storage" in first
    assert extract_domain_vocabulary([], top_n=10) == []


def test_quality_detects_numeric_obligation_overlap_and_length_warnings():
    invented = evaluate_expansion_quality(
        "The system may store audit events.",
        "The system shall store 10 audit events.",
        ["audit events"],
        0.9,
    )
    low_overlap = evaluate_expansion_quality(
        "The system shall store audit events.",
        "Unrelated payment invoice processing.",
        [],
        0.9,
    )
    too_short = evaluate_expansion_quality(
        "The system shall store audit events for operator review.",
        "store",
        [],
        0.9,
    )
    too_long = evaluate_expansion_quality(
        "The system shall store audit events.",
        "The system shall store audit events. " * 10,
        [],
        0.9,
    )

    assert invented["invented_numeric_values"] == ["10"]
    assert invented["changed_obligation_strength"] is True
    assert low_overlap["lexical_overlap"] < 0.45
    assert any("very low" in warning for warning in low_overlap["warnings"])
    assert any("too short" in warning for warning in too_short["warnings"])
    assert any("too long" in warning for warning in too_long["warnings"])


def patch_pipeline(monkeypatch):
    calls = {"base": 0, "domain": 0}

    def fake_generate(texts, **kwargs):
        calls["base"] += 1
        return np.eye(len(texts), 384, dtype=np.float32)

    def fake_domain(texts, enriched_texts=None, config=None, **kwargs):
        calls["domain"] += 1
        assert config.mode.value == "hybrid"
        assert enriched_texts[0] is not None
        return np.eye(len(texts), 384, dtype=np.float32)

    monkeypatch.setattr(pipeline, "generate_embeddings", fake_generate)
    monkeypatch.setattr(pipeline, "generate_domain_embeddings", fake_domain)
    monkeypatch.setattr(
        pipeline,
        "reduce_embeddings",
        lambda embeddings: (embeddings[:, :10], embeddings[:, :2]),
    )
    monkeypatch.setattr(
        pipeline,
        "cluster_requirements",
        lambda embeddings_10d, min_cluster_size=None, min_samples=3: (
            np.array([0, 0, 1, 1]),
            np.ones(4, dtype=np.float32),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "label_clusters",
        lambda texts, labels: {
            0: {"label": "A", "keywords": ["a"], "size": 2},
            1: {"label": "B", "keywords": ["b"], "size": 2},
        },
    )
    monkeypatch.setattr(
        pipeline,
        "build_similarity_graph",
        lambda **kwargs: {"nodes": [{"id": i} for i in range(len(kwargs["texts"]))], "edges": []},
    )
    return calls


def test_integration_helper_hands_off_to_hybrid_pipeline(monkeypatch):
    handoff = run(
        enrich_then_prepare_pipeline_inputs(
            [
                "Requirement A shall store audit logs.",
                "Requirement B shall retrieve audit logs.",
                "Requirement C shall archive audit logs.",
                "Requirement D shall delete audit logs.",
            ],
            ["A", "B", "C", "D"],
            use_cache=False,
        )
    )

    assert handoff["recommended_embedding_mode"] == "hybrid"
    assert len(handoff["enriched_texts"]) == 4
    assert "session_id" not in handoff

    calls = patch_pipeline(monkeypatch)
    result = pipeline.run_pipeline(
        texts=handoff["original_texts"],
        req_ids=handoff["requirement_ids"],
        embedding_mode="hybrid",
        enriched_texts=handoff["enriched_texts"],
    )
    base_result = pipeline.run_pipeline(
        texts=handoff["original_texts"],
        req_ids=handoff["requirement_ids"],
        embedding_mode="base",
    )

    assert result["embedding_mode"] == "hybrid"
    assert base_result["embedding_mode"] == "base"
    assert calls["domain"] == 1
    assert calls["base"] == 1
