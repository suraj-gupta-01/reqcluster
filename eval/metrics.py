"""Clustering metrics for the evaluation harness (ground truth scored in eval)."""

from __future__ import annotations

from typing import Any

import numpy as np

from reqcluster.ml.metrics import compute

# Fractions in [0, 1] from the backend -> legacy percent names used by reports/tolerances.
_PCT = {
    "purity": "accuracy_pct",  # legacy accuracy_pct was purity; baselines/floors assume that
    "accuracy_noise_inclusive": "accuracy_noise_inclusive_pct",
    "hungarian_accuracy": "hungarian_accuracy_pct",
    "noise_rate": "noise_pct",
    "coverage": "coverage_pct",
}
_PLAIN = ("ari", "nmi", "v_measure", "homogeneity", "completeness", "n_true_groups", "silhouette")


def extract_clustering_metrics(raw: dict[str, Any]) -> dict[str, Any]:
    """Map `reqcluster.ml.metrics.compute` output to the legacy metric names."""
    out: dict[str, Any] = {
        "n": raw.get("n", 0),
        "n_clusters": raw.get("n_clusters", 0),
        "noise": raw.get("noise_count", 0),
        "has_ground_truth": bool(raw.get("has_ground_truth")),
    }
    for src, dst in _PCT.items():
        if raw.get(src) is not None:
            out[dst] = raw[src] * 100
    if "accuracy_pct" in out:
        out["purity_pct"] = out["accuracy_pct"]
    for key in _PLAIN:
        if raw.get(key) is not None:
            out[key] = raw[key]
    if raw.get("confusion") is not None:
        out["confusion"] = raw["confusion"]
    return out


def score_clustering(requirements: list[dict[str, Any]], ground_truth: list[str], seed: int = 42) -> dict[str, Any]:
    """`requirements` and `ground_truth` are aligned, in upload order."""
    labels = np.array(
        [-1 if r.get("is_noise") or r.get("cluster_id") is None else int(r["cluster_id"]) for r in requirements]
    )
    return extract_clustering_metrics(compute(labels, None, seed, ground_truth=ground_truth))
