"""Clustering metrics extraction for evaluation harness."""

from __future__ import annotations

from typing import Any, Dict


def extract_clustering_metrics(raw_metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Extract standard clustering metrics from GET /api/metrics response."""
    result: Dict[str, Any] = {
        "n": raw_metrics.get("n", 0),
        "n_clusters": raw_metrics.get("n_clusters", 0),
        "noise": raw_metrics.get("noise", 0),
        "noise_pct": raw_metrics.get("noise_pct", 0.0),
        "coverage_pct": raw_metrics.get("coverage_pct", 0.0),
        "has_ground_truth": raw_metrics.get("has_ground_truth", False),
    }

    if "silhouette" in raw_metrics and raw_metrics["silhouette"] is not None:
        result["silhouette"] = raw_metrics["silhouette"]

    if raw_metrics.get("has_ground_truth"):
        for key in [
            "accuracy_pct",
            "purity_pct",
            "ari",
            "nmi",
            "v_measure",
            "homogeneity",
            "completeness",
            "n_true_groups",
            "hungarian_accuracy_pct",
        ]:
            if key in raw_metrics and raw_metrics[key] is not None:
                result[key] = raw_metrics[key]

        # Confusion matrix fields — included as structured data, not gated scalars
        for key in [
            "confusion_matrix_rows",
            "confusion_matrix_cluster_ids",
            "confusion_matrix_group_names",
        ]:
            if key in raw_metrics and raw_metrics[key] is not None:
                result[key] = raw_metrics[key]

    return result

