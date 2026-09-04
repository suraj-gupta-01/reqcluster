"""Unit tests for the dependency-tree scoring harness.

These assert the *scoring maths* on tiny synthetic inputs, independent of how
good the engine actually is — so a regression in the engine can never be masked
by a matching bug in the scorer.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
)

from core.dependency_eval import score_dependency_tree, summarize_errors


ID_TO_INDEX = {"REQ-001": 0, "REQ-002": 1, "REQ-003": 2, "REQ-004": 3}


def _edge(source, target, relation, weight=0.9, rationale=""):
    return {
        "source": source,
        "target": target,
        "relation": relation,
        "weight": weight,
        "rationale": rationale,
    }


def _gold(source_id, target_id, relation):
    return {"source_id": source_id, "target_id": target_id, "relation": relation}


def test_perfect_prediction_scores_one():
    predicted = [_edge(0, 1, "data"), _edge(1, 2, "sequential")]
    gold = [_gold("REQ-001", "REQ-002", "data"),
            _gold("REQ-002", "REQ-003", "sequential")]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)

    assert result["detection"]["precision"] == 1.0
    assert result["detection"]["recall"] == 1.0
    assert result["detection"]["f1"] == 1.0
    assert result["typed"]["f1"] == 1.0
    assert result["typed"]["true_positives"] == 2


def test_detection_and_typed_diverge_on_wrong_label():
    """The edge is found but labelled wrong: detection perfect, typed zero."""
    predicted = [_edge(0, 1, "reference")]
    gold = [_gold("REQ-001", "REQ-002", "data")]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)

    assert result["detection"]["f1"] == 1.0
    assert result["typed"]["f1"] == 0.0
    assert result["typed"]["false_positives"] == 1
    assert result["typed"]["false_negatives"] == 1


def test_undirected_reveals_inverted_edges():
    """An edge drawn backwards scores zero directed but one undirected."""
    predicted = [_edge(1, 0, "data")]
    gold = [_gold("REQ-001", "REQ-002", "data")]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)

    assert result["detection"]["f1"] == 0.0
    assert result["undirected"]["f1"] == 1.0


def test_precision_recall_arithmetic():
    # TP = 2, FP = 1, FN = 1  ->  P = R = 2/3, F1 = 2/3
    predicted = [
        _edge(0, 1, "data"),
        _edge(1, 2, "data"),
        _edge(2, 3, "data"),
    ]
    gold = [
        _gold("REQ-001", "REQ-002", "data"),
        _gold("REQ-002", "REQ-003", "data"),
        _gold("REQ-001", "REQ-004", "data"),
    ]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)

    assert result["detection"]["true_positives"] == 2
    assert result["detection"]["false_positives"] == 1
    assert result["detection"]["false_negatives"] == 1
    assert result["detection"]["precision"] == pytest.approx(0.6667, abs=1e-3)
    assert result["detection"]["recall"] == pytest.approx(0.6667, abs=1e-3)
    assert result["detection"]["f1"] == pytest.approx(0.6667, abs=1e-3)


def test_per_relation_breakdown_is_independent():
    predicted = [_edge(0, 1, "data"), _edge(1, 2, "sequential")]
    gold = [
        _gold("REQ-001", "REQ-002", "data"),
        _gold("REQ-002", "REQ-003", "hierarchical"),
    ]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)["per_relation"]

    assert result["data"]["f1"] == 1.0
    assert result["sequential"]["precision"] == 0.0
    assert result["sequential"]["support"] == 0
    assert result["hierarchical"]["recall"] == 0.0
    assert result["hierarchical"]["support"] == 1


def test_gold_edges_outside_the_dataset_are_dropped_not_counted_as_misses():
    """A gold edge naming an absent requirement cannot be predicted, so counting
    it as a false negative would understate recall for the wrong reason."""
    predicted = [_edge(0, 1, "data")]
    gold = [
        _gold("REQ-001", "REQ-002", "data"),
        _gold("REQ-999", "REQ-002", "data"),
    ]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)

    assert result["detection"]["recall"] == 1.0
    assert result["counts"]["n_gold_dropped"] == 1
    assert result["counts"]["n_gold_scored"] == 1


def test_self_loops_and_blank_gold_rows_are_dropped():
    gold = [
        _gold("REQ-001", "REQ-001", "data"),
        _gold("", "REQ-002", "data"),
        _gold("REQ-001", "REQ-002", "data"),
    ]
    result = score_dependency_tree([_edge(0, 1, "data")], gold, ID_TO_INDEX)

    assert result["counts"]["n_gold_dropped"] == 2
    assert result["detection"]["f1"] == 1.0


def test_unmappable_predicted_edges_are_reported_not_crashing():
    predicted = [_edge(0, 1, "data"), _edge(0, 99, "data")]
    gold = [_gold("REQ-001", "REQ-002", "data")]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)

    assert result["counts"]["n_predicted_unmappable"] == 1
    assert result["detection"]["precision"] == 1.0


def test_empty_inputs_do_not_raise():
    result = score_dependency_tree([], [], {})
    assert result["detection"]["f1"] == 0.0
    assert result["typed"]["support"] == 0
    assert result["per_relation"] == {}


def test_no_predictions_scores_zero_recall():
    gold = [_gold("REQ-001", "REQ-002", "data")]
    result = score_dependency_tree([], gold, ID_TO_INDEX)

    assert result["detection"]["recall"] == 0.0
    assert result["detection"]["false_negatives"] == 1


def test_duplicate_predicted_edges_are_deduplicated():
    predicted = [_edge(0, 1, "data"), _edge(0, 1, "data")]
    gold = [_gold("REQ-001", "REQ-002", "data")]

    result = score_dependency_tree(predicted, gold, ID_TO_INDEX)

    assert result["detection"]["precision"] == 1.0
    assert result["detection"]["true_positives"] == 1


def test_relation_labels_are_case_and_whitespace_insensitive():
    predicted = [_edge(0, 1, "  DATA ")]
    gold = [_gold("REQ-001", "REQ-002", "Data")]

    assert score_dependency_tree(predicted, gold, ID_TO_INDEX)["typed"]["f1"] == 1.0


def test_summarize_errors_classifies_each_failure_mode():
    predicted = [
        _edge(0, 1, "reference"),   # in gold, wrong label -> mislabeled
        _edge(2, 3, "data"),        # not in gold          -> spurious
    ]
    gold = [
        _gold("REQ-001", "REQ-002", "data"),
        _gold("REQ-002", "REQ-003", "sequential"),  # not predicted -> missed
    ]

    errors = summarize_errors(predicted, gold, ID_TO_INDEX)

    assert errors["totals"] == {"missed": 1, "spurious": 1, "mislabeled": 1}
    assert errors["missed"][0]["source_id"] == "REQ-002"
    assert errors["spurious"][0]["source_id"] == "REQ-003"
    assert errors["mislabeled"][0]["predicted_relation"] == "reference"
    assert errors["mislabeled"][0]["gold_relation"] == "data"


def test_summarize_errors_respects_limit():
    predicted = []
    gold = [
        _gold("REQ-001", "REQ-002", "data"),
        _gold("REQ-002", "REQ-003", "data"),
        _gold("REQ-003", "REQ-004", "data"),
    ]

    errors = summarize_errors(predicted, gold, ID_TO_INDEX, limit=2)

    assert len(errors["missed"]) == 2
    assert errors["totals"]["missed"] == 3


# ---------------------------------------------------------------------------
# Objective vs judgment-based gold edges
# ---------------------------------------------------------------------------

def test_partition_gold_by_objectivity_splits_on_literal_citation():
    from core.dependency_eval import partition_gold_by_objectivity

    texts = [
        "The power unit shall provide a regulated bus.",
        "The avionics shall require the regulated bus defined in REQ-001.",
        "The logger shall record faults.",
    ]
    req_ids = ["REQ-001", "REQ-002", "REQ-003"]
    gold = [
        _gold("REQ-001", "REQ-002", "data"),   # REQ-002's text cites REQ-001
        _gold("REQ-001", "REQ-003", "data"),   # inferred by a human, not cited
    ]

    objective, judgment = partition_gold_by_objectivity(gold, texts, req_ids)

    assert [e["target_id"] for e in objective] == ["REQ-002"]
    assert [e["target_id"] for e in judgment] == ["REQ-003"]


def test_partition_ignores_self_citations_and_unknown_ids():
    from core.dependency_eval import partition_gold_by_objectivity

    texts = [
        "REQ-001 shall reference itself and also REQ-999.",
        "The second requirement stands alone.",
    ]
    req_ids = ["REQ-001", "REQ-002"]
    gold = [_gold("REQ-001", "REQ-002", "data")]

    objective, judgment = partition_gold_by_objectivity(gold, texts, req_ids)

    assert objective == []
    assert len(judgment) == 1


def test_partition_matches_ids_across_separator_styles():
    from core.dependency_eval import partition_gold_by_objectivity

    texts = ["Base requirement.", "Derived from REQ_001 upstream."]
    gold = [_gold("REQ-001", "REQ-002", "data")]

    objective, _ = partition_gold_by_objectivity(gold, texts, ["REQ-001", "REQ-002"])

    assert len(objective) == 1


def test_partition_tolerates_more_texts_than_ids():
    from core.dependency_eval import partition_gold_by_objectivity

    objective, judgment = partition_gold_by_objectivity(
        [_gold("REQ-001", "REQ-002", "data")],
        ["a", "cites REQ-001", "extra text with no id"],
        ["REQ-001", "REQ-002"],
    )
    assert len(objective) == 1
    assert judgment == []
