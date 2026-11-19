"""Refinement suggestion judgment using ground-truth module labels."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from typing import Any


def build_cluster_label_distribution(
    requirements: list[dict[str, Any]],
) -> dict[int, Counter]:
    """Map each non-noise cluster_id to its ground-truth module distribution."""
    cluster_dist: dict[int, Counter] = {}
    for r in requirements:
        cid = r.get("cluster_id")
        if cid is None or int(cid) == -1:
            continue
        cid = int(cid)
        module = str(r.get("module") or "").strip()
        if not module:
            continue
        cluster_dist.setdefault(cid, Counter())[module] += 1
    return cluster_dist


def evaluate_refinement_metrics(
    requirements: list[dict[str, Any]],
    suggestions_data: dict[str, Any] | list[dict[str, Any]],
    impurity_threshold: float = 0.90,
) -> dict[str, Any]:
    """Score merge/split refinement suggestions against ground truth labels."""
    cluster_dist = build_cluster_label_distribution(requirements)
    if not cluster_dist:
        return {
            "n_merge_suggestions": 0,
            "n_split_suggestions": 0,
            "merge_precision": 1.0,
            "split_precision": 1.0,
            "suggestion_precision": 1.0,
        }

    # Extract dominant labels and impurity status
    dominant_labels: dict[int, str] = {}
    impure_clusters: set[int] = set()

    for cid, counter in cluster_dist.items():
        total = sum(counter.values())
        if total == 0:
            continue
        top_label, top_count = counter.most_common(1)[0]
        dominant_labels[cid] = top_label
        purity = top_count / total
        if purity < impurity_threshold:
            impure_clusters.add(cid)

    # Find all eligible merge pairs (sharing dominant label)
    eligible_merge_pairs: set[tuple[int, int]] = set()
    clusters = sorted(dominant_labels.keys())
    for c1, c2 in combinations(clusters, 2):
        if dominant_labels[c1] == dominant_labels[c2]:
            eligible_merge_pairs.add((c1, c2))

    # Parse suggestions
    merge_suggestions: list[dict[str, Any]] = []
    split_suggestions: list[dict[str, Any]] = []

    if isinstance(suggestions_data, list):
        for s in suggestions_data:
            stype = str(s.get("suggestion_type") or "").lower()
            if stype == "merge":
                merge_suggestions.append(s)
            elif stype == "split":
                split_suggestions.append(s)
    elif isinstance(suggestions_data, dict):
        merge_suggestions = suggestions_data.get("merge_suggestions") or []
        split_suggestions = suggestions_data.get("split_suggestions") or []

    # Score merges
    correct_merges = 0
    for s in merge_suggestions:
        c_a = s.get("cluster_a_id")
        c_b = s.get("cluster_b_id")
        if c_a is not None and c_b is not None:
            ca_int, cb_int = int(c_a), int(c_b)
            if (
                ca_int in dominant_labels
                and cb_int in dominant_labels
                and dominant_labels[ca_int] == dominant_labels[cb_int]
            ):
                correct_merges += 1

    # Score splits
    correct_splits = 0
    for s in split_suggestions:
        cid = s.get("cluster_id")
        if cid is not None:
            cid_int = int(cid)
            if cid_int in impure_clusters:
                correct_splits += 1

    n_merges = len(merge_suggestions)
    n_splits = len(split_suggestions)
    total_suggestions = n_merges + n_splits

    merge_precision = round(correct_merges / n_merges, 4) if n_merges > 0 else 1.0
    merge_recall = (
        round(correct_merges / len(eligible_merge_pairs), 4)
        if len(eligible_merge_pairs) > 0
        else 1.0
    )

    split_precision = round(correct_splits / n_splits, 4) if n_splits > 0 else 1.0
    split_recall = (
        round(correct_splits / len(impure_clusters), 4)
        if len(impure_clusters) > 0
        else 1.0
    )

    suggestion_precision = (
        round((correct_merges + correct_splits) / total_suggestions, 4)
        if total_suggestions > 0
        else 1.0
    )

    return {
        "n_merge_suggestions": n_merges,
        "correct_merges": correct_merges,
        "n_eligible_merge_pairs": len(eligible_merge_pairs),
        "merge_precision": merge_precision,
        "merge_recall": merge_recall,
        "n_split_suggestions": n_splits,
        "correct_splits": correct_splits,
        "n_impure_clusters": len(impure_clusters),
        "split_precision": split_precision,
        "split_recall": split_recall,
        "suggestion_precision": suggestion_precision,
    }
