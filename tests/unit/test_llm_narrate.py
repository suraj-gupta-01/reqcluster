import pytest

from reqcluster.config import get_settings
from reqcluster.llm.gateway import LLMGateway
from reqcluster.llm.narrate import (
    MAX_EXPLANATION_CHARS,
    cluster_narrative,
    cluster_narratives,
    cluster_summary,
    explain_edge,
    merge_rationale,
    mock_explanation,
    parse_explanation_response,
    split_rationale,
)

ART = {"pass": "pass_0_artifact_match", "artifact": "temperature reading",
       "producer_artifact_name": "temperature reading"}


def test_mock_narratives_are_templates():
    gw = LLMGateway(get_settings())
    r = cluster_narrative(gw, "merge", cluster_a_label="Thermal", cluster_b_label="Power",
                          centroid_similarity=0.8, silhouette_delta=0.01,
                          coherence_a=0.7, coherence_b=0.6)
    assert r.source == "fallback" and "Thermal" in r.output and "0.0100" in r.output
    s = cluster_narrative(gw, "summary", cluster_label="Thermal", keywords=["temperature"],
                          representative_texts=["The fan shall run."])
    assert s.output == cluster_summary("Thermal", ["temperature"], ["The fan shall run."])


def test_merge_split_branches():
    assert "neutral" in merge_rationale("A", "B", 0.9, 0, 0.9, 0.9)
    assert "decreases" in merge_rationale("A", "B", 0.9, -0.1, 0.1, 0.1)
    assert "5 and 4" in split_rationale("A", 0.6, 120.0, 0.4, [5, 4])
    assert "high internal spread" not in split_rationale("A", 0.6, 1.0, 0.1, [1])


def test_summary_without_keywords():
    assert "contains related requirements" in cluster_summary("X", [], [])


def test_mock_explanation_deterministic_and_cites_artifact():
    a = mock_explanation("REQ-001", "REQ-002", "data", ART, "High")
    assert a == mock_explanation("REQ-001", "REQ-002", "data", ART, "High")
    assert "temperature reading" in a and "high" in a.lower()


def test_mock_explanation_synonym_and_passes():
    ev = {**ART, "consumer_artifact_name": "temp value", "synonym_match": True,
          "artifact_similarity": 0.84}
    t = mock_explanation("R1", "R2", "data", ev, "High")
    assert "synonym" in t and "0.84" in t
    for ev, exp in [({"pass": "pass_1_explicit_reference", "matched_text": "REQ-001"}, "REQ-001"),
                    ({"pass": "pass_2_hierarchical", "cosine_similarity": 0.9}, "cluster"),
                    ({"pass": "pass_2_sequential", "cue": "after"}, "after")]:
        assert exp in mock_explanation("R1", "R2", "data", ev, "Medium")


def test_mock_explanation_empty_evidence():
    t = mock_explanation("R1", "R2", "data", {}, "Low", rationale="Fallback rationale text.")
    assert "R1" in t and "Fallback rationale" in t


def test_explain_edge_mock_provider():
    r = explain_edge(LLMGateway(get_settings()), "a", "b", "data", ART, source_id="R1", target_id="R2")
    assert r.source == "fallback" and "R1" in r.output


@pytest.mark.parametrize("raw,frag", [
    ("A plain paragraph.", "A plain paragraph."),
    ({"explanation": "From a dict."}, "From a dict."),
    ({"content": "Under content."}, "Under content."),
    (["First element."], "First element."),
    ("```\nFenced prose.\n```", "Fenced prose."),
    ("```json\nStill prose.\n```", "Still prose."),
    ("  Padded   with\n\nwhitespace.  ", "Padded with whitespace."),
])
def test_parser_coerces_shapes(raw, frag):
    assert frag in parse_explanation_response(raw)


@pytest.mark.parametrize("raw", [None, "", "   ", [], {"explanation": ""}])
def test_parser_rejects_empty(raw):
    with pytest.raises(ValueError):
        parse_explanation_response(raw)


def test_parser_bounds_length():
    assert len(parse_explanation_response("word " * 5000)) <= MAX_EXPLANATION_CHARS


def test_cluster_narratives_match_single_calls_in_order():
    g = LLMGateway(get_settings())
    items = [{"kind": "summary", "cluster_label": "A", "keywords": ["x"], "representative_texts": []},
             {"kind": "summary", "cluster_label": "B", "keywords": [], "representative_texts": []}]
    got = cluster_narratives(g, items)
    want = [cluster_narrative(g, **i) for i in items]
    assert [r.output for r in got] == [r.output for r in want]
    assert "A" in got[0].output and "B" in got[1].output
