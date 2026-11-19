"""Dependency tree precision and recall evaluation against ground truth."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any


_REQ_ID_PATTERN = re.compile(r"\b(?:REQ|SR|FR|SYS|R)[-_ ]?\d{1,5}\b", re.IGNORECASE)


def extract_expected_reference_edges(
    requirements: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Construct expected reference edges from requirement text mentions.

    Returns a set of (source_req_id, target_req_id) pairs where source is mentioned
    in target's text (meaning source is a prerequisite/reference for target).
    """
    id_map: dict[str, str] = {}
    for r in requirements:
        req_id = str(r.get("req_id") or "").strip()
        if req_id:
            id_map[req_id.upper()] = req_id
            # Also map normalized without hyphens/underscores/spaces
            norm = re.sub(r"[-_ ]", "", req_id.upper())
            id_map[norm] = req_id

    expected_edges: set[tuple[str, str]] = set()

    for r in requirements:
        target_id = str(r.get("req_id") or "").strip()
        if not target_id:
            continue

        text = str(r.get("text") or "")
        matches = _REQ_ID_PATTERN.findall(text)
        for match in matches:
            match_upper = match.upper()
            match_norm = re.sub(r"[-_ ]", "", match_upper)
            source_id = id_map.get(match_upper) or id_map.get(match_norm)
            if source_id and source_id != target_id:
                expected_edges.add((source_id, target_id))

    return expected_edges


def load_expected_data_edges(edges_path: Path | str | None) -> set[tuple[str, str]]:
    """Load ground-truth data/flow edges from an optional .edges.json file."""
    if not edges_path:
        return set()

    path = Path(edges_path)
    if not path.exists():
        return set()

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            edges: set[tuple[str, str]] = set()
            for item in data:
                src = str(item.get("source") or "").strip()
                tgt = str(item.get("target") or "").strip()
                if src and tgt:
                    edges.add((src, tgt))
            return edges
    except Exception:
        return set()


def extract_inferred_edges(
    dep_response: dict[str, Any],
) -> tuple[set[tuple[str, str]], set[tuple[str, str]]]:
    """Extract inferred reference and data/other edges from GET /api/dependencies."""
    inferred_ref: set[tuple[str, str]] = set()
    inferred_data: set[tuple[str, str]] = set()

    # Check rationale dependencies
    rationale = dep_response.get("rationale") or {}
    deps_list = rationale.get("dependencies") or []
    for dep in deps_list:
        src = str(dep.get("source_req_id") or "").strip()
        tgt = str(dep.get("target_req_id") or "").strip()
        rel = str(dep.get("relation") or "").strip().lower()
        if src and tgt:
            if rel == "reference":
                inferred_ref.add((src, tgt))
            else:
                inferred_data.add((src, tgt))

    # Check direct edges list if present
    edges_list = dep_response.get("edges") or []
    for edge in edges_list:
        src = str(edge.get("source") or "").strip()
        tgt = str(edge.get("target") or "").strip()
        rel = str(edge.get("relation") or "").strip().lower()
        if src and tgt:
            if rel == "reference":
                inferred_ref.add((src, tgt))
            else:
                inferred_data.add((src, tgt))

    return inferred_ref, inferred_data


def compute_precision_recall_f1(
    inferred: set[tuple[str, str]],
    expected: set[tuple[str, str]],
) -> dict[str, float]:
    """Compute precision, recall, and F1 given inferred and expected edge sets."""
    if not expected:
        return {
            "precision": 1.0 if not inferred else 0.0,
            "recall": 1.0,
            "f1": 1.0 if not inferred else 0.0,
        }

    if not inferred:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    tp = len(inferred & expected)
    precision = tp / len(inferred)
    recall = tp / len(expected)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_dependency_metrics(
    requirements: list[dict[str, Any]],
    dep_response: dict[str, Any],
    edges_path: Path | str | None = None,
) -> dict[str, Any]:
    """Compute precision/recall/F1 for reference and data edges."""
    expected_ref = extract_expected_reference_edges(requirements)
    expected_data = load_expected_data_edges(edges_path)

    inferred_ref, inferred_data = extract_inferred_edges(dep_response)

    ref_scores = compute_precision_recall_f1(inferred_ref, expected_ref)

    result: dict[str, Any] = {
        "reference_precision": ref_scores["precision"],
        "reference_recall": ref_scores["recall"],
        "reference_f1": ref_scores["f1"],
        "n_expected_reference_edges": len(expected_ref),
        "n_inferred_reference_edges": len(inferred_ref),
        "n_inferred_total_edges": len(inferred_ref) + len(inferred_data),
    }

    if expected_data:
        data_scores = compute_precision_recall_f1(inferred_data, expected_data)
        result.update({
            "data_precision": data_scores["precision"],
            "data_recall": data_scores["recall"],
            "data_f1": data_scores["f1"],
            "n_expected_data_edges": len(expected_data),
            "n_inferred_data_edges": len(inferred_data),
        })

    return result
