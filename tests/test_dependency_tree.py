"""Tests for the DP5 dependency tree inference engine."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend")))

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
