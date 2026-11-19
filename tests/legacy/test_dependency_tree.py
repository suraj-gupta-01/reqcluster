"""Tests for the DP5 dependency tree inference engine."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend")))

from core.dependency_tree import (
    DATA,
    HIERARCHICAL,
    REFERENCE,
    SEQUENTIAL,
    build_dependency_tree,
    _assign_levels,
    _break_cycles,
    DependencyEdge,
)


def _embeddings_for(texts):
    """Deterministic embeddings: similar texts share leading dimensions."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal((len(texts), 384)).astype(np.float32)
    norms = np.linalg.norm(base, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return base / norms


def test_explicit_reference_creates_directed_edge():
    texts = [
        "The system shall expose a thermal limit value.",
        "The cooling controller shall act as defined in REQ-001.",
    ]
    req_ids = ["REQ-001", "REQ-002"]
    labels = np.array([0, 0], dtype=np.int32)
    emb = _embeddings_for(texts)

    tree = build_dependency_tree(emb, texts, req_ids, labels, sim_threshold=0.0)
    refs = [e for e in tree["edges"] if e["relation"] in (REFERENCE, HIERARCHICAL)]
    assert any(e["source"] == 0 and e["target"] == 1 for e in refs), tree["edges"]
    # REQ-002 references REQ-001 => REQ-001 is the prerequisite (source).
    edge = next(e for e in refs if e["target"] == 1)
    assert edge["weight"] >= 0.9


def test_data_dependency_producer_to_consumer():
    texts = [
        "The sensor module shall generate a temperature reading every second.",
        "The controller shall use the temperature reading to adjust fan speed.",
    ]
    req_ids = ["REQ-010", "REQ-011"]
    labels = np.array([0, 0], dtype=np.int32)
    # Force high similarity so the pair is examined.
    emb = np.ones((2, 384), dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

    tree = build_dependency_tree(emb, texts, req_ids, labels, sim_threshold=0.1)
    data_edges = [e for e in tree["edges"] if e["relation"] == DATA]
    assert any(e["source"] == 0 and e["target"] == 1 for e in data_edges), tree["edges"]


def test_output_is_acyclic_and_levels_consistent():
    texts = [f"The system shall handle function {i} using shared data bus." for i in range(12)]
    req_ids = [f"REQ-{i + 1:03d}" for i in range(12)]
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int32)
    emb = _embeddings_for(texts)

    tree = build_dependency_tree(emb, texts, req_ids, labels)
    # Levels are present and non-negative.
    levels = {n["id"]: n["level"] for n in tree["nodes"]}
    assert all(v >= 0 for v in levels.values())
    # Every edge goes from a lower or equal... strictly lower level (DAG property).
    for e in tree["edges"]:
        assert levels[e["source"]] < levels[e["target"]]


def test_break_cycles_removes_back_edges():
    # 0->1, 1->2, 2->0 would be a cycle; the lowest-weight edge is dropped.
    edges = [
        DependencyEdge(0, 1, DATA, 0.9, ""),
        DependencyEdge(1, 2, DATA, 0.8, ""),
        DependencyEdge(2, 0, DATA, 0.5, ""),
    ]
    kept = _break_cycles(3, edges)
    assert len(kept) == 2
    levels = _assign_levels(3, kept)
    assert max(levels) >= 1


def test_empty_and_single_requirement():
    emb = np.zeros((1, 384), dtype=np.float32)
    tree = build_dependency_tree(emb, ["only one requirement here"], ["REQ-001"], np.array([0]))
    assert tree["edges"] == []
    assert len(tree["nodes"]) == 1
    assert tree["stats"]["n_edges"] == 0


def test_max_edges_cap():
    texts = [f"The system shall provide and use data field {i}." for i in range(40)]
    req_ids = [f"REQ-{i + 1:03d}" for i in range(40)]
    labels = np.zeros(40, dtype=np.int32)
    emb = np.ones((40, 384), dtype=np.float32)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    tree = build_dependency_tree(emb, texts, req_ids, labels, sim_threshold=0.1, max_edges=5)
    assert len(tree["edges"]) <= 5


def _similar_embeddings(count):
    """Near-identical unit vectors, so every pair clears the similarity gate.

    ``_embeddings_for`` returns random directions whose cosine can be slightly
    negative; a pair is then never examined and a Pass 2 assertion would pass
    vacuously.
    """
    base = np.ones((count, 384), dtype=np.float32)
    for i in range(count):
        base[i, i % 384] += 0.05 * (i + 1)
    return base / np.linalg.norm(base, axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Sequential trigger-clause gating
# ---------------------------------------------------------------------------

def test_sequential_needs_shared_term_inside_the_trigger_clause():
    """A cue alone is not evidence of a link to the *other* requirement.

    Requirements written in EARS style ("WHEN <trigger>, the software shall
    <action>") almost all carry a cue, so firing on cue presence plus any shared
    word anywhere makes nearly every similar pair sequential.
    """
    texts = [
        "The software shall calibrate the vibration sensor during manufacture.",
        "WHEN the operator presses the start button, the software shall log the "
        "vibration sensor reading.",
    ]
    req_ids = ["REQ-001", "REQ-002"]
    labels = np.array([0, 0], dtype=np.int32)

    tree = build_dependency_tree(
        _similar_embeddings(len(texts)), texts, req_ids, labels, sim_threshold=0.0
    )

    # "vibration sensor" is shared, but only in the action clause — the trigger
    # is about the operator's button, which REQ-001 says nothing about.
    #
    # This is not vacuous: the sibling test below uses the same embeddings and
    # threshold, so the pair is definitely examined; only the trigger differs.
    assert not [e for e in tree["edges"] if e["relation"] == SEQUENTIAL], tree["edges"]


def test_sequential_fires_when_the_trigger_clause_shares_a_term():
    texts = [
        "The software shall complete the power-on selftest during initialization.",
        "WHEN the power-on selftest completes, the software shall enable the "
        "application layer.",
    ]
    req_ids = ["REQ-001", "REQ-002"]
    labels = np.array([0, 0], dtype=np.int32)

    tree = build_dependency_tree(
        _similar_embeddings(len(texts)), texts, req_ids, labels, sim_threshold=0.0
    )

    seq = [e for e in tree["edges"] if e["relation"] == SEQUENTIAL]
    assert seq, tree["edges"]
    edge = seq[0]
    assert edge["source"] == 0 and edge["target"] == 1
    assert "selftest" in edge["evidence"]["trigger_shared_tokens"]
    assert edge["evidence"]["cue_clause"].startswith("when the power-on selftest")


def test_cue_clause_stops_at_the_first_clause_boundary():
    from core.dependency_tree import _cue_clause

    clause = _cue_clause(
        "when the power-on test completes, the software shall enable outputs", "when"
    )
    assert clause == "when the power-on test completes"


def test_cue_clause_is_bounded_without_punctuation():
    from core.dependency_tree import _CUE_CLAUSE_MAX_WORDS, _cue_clause

    clause = _cue_clause("once " + " ".join(f"word{i}" for i in range(40)), "once")
    assert len(clause.split()) == _CUE_CLAUSE_MAX_WORDS


# ---------------------------------------------------------------------------
# Cross-pass corroboration
# ---------------------------------------------------------------------------

def test_artifact_flow_supersedes_a_bare_reference_label():
    """`reference` says only "an identifier was mentioned".

    When another pass finds a concrete relation for the same pair, that relation
    is the informative one; the citation is retained as corroborating evidence
    and keeps its high weight.
    """
    texts = [
        "The power unit shall provide the regulated voltage bus.",
        "The avionics shall require the regulated voltage bus defined in REQ-001.",
    ]
    req_ids = ["REQ-001", "REQ-002"]
    labels = np.array([0, 0], dtype=np.int32)

    tree = build_dependency_tree(
        _embeddings_for(texts), texts, req_ids, labels, sim_threshold=0.0
    )

    edge = next(e for e in tree["edges"] if e["source"] == 0 and e["target"] == 1)
    assert edge["relation"] == DATA
    assert edge["weight"] >= 0.95  # keeps the explicit citation's confidence
    assert "pass_1_explicit_reference" in edge["evidence"]["corroborating_passes"]
    assert edge["evidence"]["corroboration"]["pass_1_explicit_reference"]["relation"] == REFERENCE
    assert "explicitly" in edge["rationale"]


def test_reference_survives_when_no_other_pass_fires():
    texts = [
        "The system shall expose a thermal limit value.",
        "The cooling controller shall act as defined in REQ-001.",
    ]
    req_ids = ["REQ-001", "REQ-002"]
    labels = np.array([0, 0], dtype=np.int32)

    tree = build_dependency_tree(
        _embeddings_for(texts), texts, req_ids, labels, sim_threshold=0.0
    )
    edge = next(e for e in tree["edges"] if e["source"] == 0 and e["target"] == 1)
    assert edge["relation"] in (REFERENCE, HIERARCHICAL)
    assert "corroborating_passes" not in edge["evidence"]


def test_corroboration_is_order_independent():
    """Pass 0 and Pass 1 must settle the same way regardless of which ran first."""
    from core.dependency_tree import DATA as _DATA, REFERENCE as _REF, _add_edge

    art = {"pass": "pass_0_artifact_match", "artifact": "bus"}
    ref = {"pass": "pass_1_explicit_reference", "matched_text": "REQ-001"}

    forward = {}
    _add_edge(forward, 0, 1, _REF, 0.95, "cited", dict(ref))
    _add_edge(forward, 0, 1, _DATA, 0.90, "artifact", dict(art))

    backward = {}
    _add_edge(backward, 0, 1, _DATA, 0.90, "artifact", dict(art))
    _add_edge(backward, 0, 1, _REF, 0.95, "cited", dict(ref))

    a, b = forward[(0, 1)], backward[(0, 1)]
    assert a.relation == b.relation == _DATA
    assert a.weight == b.weight == 0.95
    assert a.evidence["corroborating_passes"] == b.evidence["corroborating_passes"]
