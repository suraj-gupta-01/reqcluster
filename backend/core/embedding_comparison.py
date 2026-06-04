from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


DEFAULT_DELTA_THRESHOLDS = (0.05, 0.10, 0.20)
DEFAULT_NEIGHBOR_K = 5
MAX_NEIGHBOR_ITEMS = 2_000


def _safe_float(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not np.isfinite(value):
        return None
    return round(value, digits)


def _empty_report(warnings: list[str]) -> Dict[str, Any]:
    return {
        "n_requirements": 0,
        "per_requirement": [],
        "aggregate": {
            "mean_cosine_similarity": None,
            "median_cosine_similarity": None,
            "min_cosine_similarity": None,
            "max_cosine_similarity": None,
            "mean_delta": None,
        },
        "delta_threshold_counts": {f"{t:.2f}": 0 for t in DEFAULT_DELTA_THRESHOLDS},
        "nearest_neighbor_preservation": {
            "k": 0,
            "score": None,
            "evaluated_count": 0,
            "max_items": MAX_NEIGHBOR_ITEMS,
            "truncated": False,
        },
        "cluster_impact": {},
        "warnings": warnings,
    }


def _as_matrix(value: Any, name: str, warnings: list[str]) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except Exception:
        warnings.append(f"{name} could not be converted to a numeric matrix.")
        return None

    if arr.ndim != 2:
        warnings.append(f"{name} must be a 2D matrix.")
        return None
    if arr.shape[1] == 0:
        warnings.append(f"{name} must have at least one embedding dimension.")
        return None
    if not np.all(np.isfinite(arr)):
        warnings.append(f"{name} contained NaN or infinite values; invalid values were replaced.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def _row_normalize(arr: np.ndarray, name: str, warnings: list[str]) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    zero_rows = norms[:, 0] <= 1e-12
    if np.any(zero_rows):
        warnings.append(f"{name} contained zero-norm rows; cosine values for them may be 0.")
    safe_norms = norms.copy()
    safe_norms[zero_rows] = 1.0
    return arr / safe_norms


def _topk_neighbors(similarity: np.ndarray, k: int) -> list[set[int]]:
    neighbors: list[set[int]] = []
    for idx in range(similarity.shape[0]):
        ordered = np.argsort(-similarity[idx], kind="mergesort")
        neighbors.append(set(int(i) for i in ordered[:k]))
    return neighbors


def _nearest_neighbor_preservation(
    base_norm: np.ndarray,
    candidate_norm: np.ndarray,
    k: int,
    max_items: int,
    warnings: list[str],
) -> Dict[str, Any]:
    n = base_norm.shape[0]
    if n <= 1:
        warnings.append("Nearest-neighbor preservation requires at least two requirements.")
        return {
            "k": 0,
            "score": None,
            "evaluated_count": n,
            "max_items": max_items,
            "truncated": False,
        }

    evaluated_count = min(n, max_items)
    truncated = evaluated_count < n
    if truncated:
        warnings.append(
            f"Nearest-neighbor preservation evaluated the first {evaluated_count} requirements."
        )

    actual_k = min(max(1, int(k)), evaluated_count - 1)
    base_subset = base_norm[:evaluated_count]
    candidate_subset = candidate_norm[:evaluated_count]

    base_sim = base_subset @ base_subset.T
    candidate_sim = candidate_subset @ candidate_subset.T
    np.fill_diagonal(base_sim, -np.inf)
    np.fill_diagonal(candidate_sim, -np.inf)

    base_neighbors = _topk_neighbors(base_sim, actual_k)
    candidate_neighbors = _topk_neighbors(candidate_sim, actual_k)

    overlaps = []
    for base_set, candidate_set in zip(base_neighbors, candidate_neighbors):
        union = base_set | candidate_set
        overlaps.append(1.0 if not union else len(base_set & candidate_set) / len(union))

    return {
        "k": actual_k,
        "score": _safe_float(float(np.mean(overlaps)) if overlaps else None),
        "evaluated_count": evaluated_count,
        "max_items": max_items,
        "truncated": truncated,
    }


def _labels_array(
    labels: Sequence[int] | np.ndarray | None,
    expected_rows: int,
    name: str,
    warnings: list[str],
) -> np.ndarray | None:
    if labels is None:
        return None
    arr = np.asarray(labels)
    if arr.ndim != 1 or arr.shape[0] != expected_rows:
        warnings.append(f"{name} must be a 1D label array matching the embeddings row count.")
        return None
    return arr.astype(int, copy=False)


def _silhouette_or_none(
    embeddings: np.ndarray | None,
    labels: np.ndarray | None,
    name: str,
    warnings: list[str],
) -> float | None:
    if embeddings is None or labels is None:
        return None
    if embeddings.ndim != 2 or embeddings.shape[0] != labels.shape[0]:
        warnings.append(f"{name} silhouette skipped because embeddings and labels do not align.")
        return None

    mask = labels != -1
    if int(mask.sum()) < 3:
        warnings.append(f"{name} silhouette skipped because too few non-noise points exist.")
        return None

    filtered_labels = labels[mask]
    unique_labels = set(int(v) for v in filtered_labels)
    if len(unique_labels) < 2:
        warnings.append(f"{name} silhouette skipped because fewer than two clusters exist.")
        return None
    if len(unique_labels) >= filtered_labels.shape[0]:
        warnings.append(f"{name} silhouette skipped because every point is its own cluster.")
        return None

    try:
        return float(silhouette_score(embeddings[mask], filtered_labels))
    except Exception:
        warnings.append(f"{name} silhouette calculation failed.")
        return None


def _cluster_impact(
    labels_base: Sequence[int] | np.ndarray | None,
    labels_candidate: Sequence[int] | np.ndarray | None,
    embeddings_10d_base: np.ndarray | None,
    embeddings_10d_candidate: np.ndarray | None,
    expected_rows: int,
    warnings: list[str],
) -> Dict[str, Any]:
    if labels_base is None and labels_candidate is None:
        return {}

    result: Dict[str, Any] = {
        "adjusted_rand_index": None,
        "normalized_mutual_info": None,
        "base_silhouette_score_10d": None,
        "candidate_silhouette_score_10d": None,
        "silhouette_score_delta": None,
    }

    base_labels = _labels_array(labels_base, expected_rows, "labels_base", warnings)
    candidate_labels = _labels_array(
        labels_candidate,
        expected_rows,
        "labels_candidate",
        warnings,
    )
    if base_labels is None or candidate_labels is None:
        return result

    try:
        result["adjusted_rand_index"] = _safe_float(adjusted_rand_score(base_labels, candidate_labels))
    except Exception:
        warnings.append("Adjusted Rand index calculation failed.")
    try:
        result["normalized_mutual_info"] = _safe_float(
            normalized_mutual_info_score(base_labels, candidate_labels)
        )
    except Exception:
        warnings.append("Normalized mutual information calculation failed.")

    base_silhouette = _silhouette_or_none(
        embeddings_10d_base,
        base_labels,
        "base",
        warnings,
    )
    candidate_silhouette = _silhouette_or_none(
        embeddings_10d_candidate,
        candidate_labels,
        "candidate",
        warnings,
    )
    result["base_silhouette_score_10d"] = _safe_float(base_silhouette)
    result["candidate_silhouette_score_10d"] = _safe_float(candidate_silhouette)
    if base_silhouette is not None and candidate_silhouette is not None:
        result["silhouette_score_delta"] = _safe_float(candidate_silhouette - base_silhouette)

    return result


def compare_embeddings(
    base_embeddings: np.ndarray,
    candidate_embeddings: np.ndarray,
    labels_base: Sequence[int] | np.ndarray | None = None,
    labels_candidate: Sequence[int] | np.ndarray | None = None,
    embeddings_10d_base: np.ndarray | None = None,
    embeddings_10d_candidate: np.ndarray | None = None,
    k: int = DEFAULT_NEIGHBOR_K,
    delta_thresholds: Sequence[float] = DEFAULT_DELTA_THRESHOLDS,
    max_neighbor_items: int = MAX_NEIGHBOR_ITEMS,
) -> Dict[str, Any]:
    """
    Compare base embeddings against enriched or hybrid embeddings.

    The result is JSON-serializable and includes warnings instead of raising
    for metric-specific edge cases.
    """
    warnings: list[str] = []
    base = _as_matrix(base_embeddings, "base_embeddings", warnings)
    candidate = _as_matrix(candidate_embeddings, "candidate_embeddings", warnings)
    if base is None or candidate is None:
        return _empty_report(warnings)
    if base.shape != candidate.shape:
        warnings.append("base_embeddings and candidate_embeddings must have the same shape.")
        return _empty_report(warnings)

    n = base.shape[0]
    thresholds = tuple(float(t) for t in delta_thresholds)
    if n == 0:
        report = _empty_report(warnings)
        report["delta_threshold_counts"] = {f"{t:.2f}": 0 for t in thresholds}
        return report

    base_norm = _row_normalize(base, "base_embeddings", warnings)
    candidate_norm = _row_normalize(candidate, "candidate_embeddings", warnings)
    cosine_similarities = np.clip(np.sum(base_norm * candidate_norm, axis=1), -1.0, 1.0)
    deltas = 1.0 - cosine_similarities

    per_requirement = [
        {
            "index": int(idx),
            "cosine_similarity": _safe_float(float(sim)),
            "cosine_distance_delta": _safe_float(float(delta)),
        }
        for idx, (sim, delta) in enumerate(zip(cosine_similarities, deltas))
    ]

    nearest_neighbor = _nearest_neighbor_preservation(
        base_norm,
        candidate_norm,
        k=k,
        max_items=max_neighbor_items,
        warnings=warnings,
    )

    cluster_impact = _cluster_impact(
        labels_base,
        labels_candidate,
        embeddings_10d_base,
        embeddings_10d_candidate,
        n,
        warnings,
    )

    return {
        "n_requirements": int(n),
        "per_requirement": per_requirement,
        "aggregate": {
            "mean_cosine_similarity": _safe_float(float(np.mean(cosine_similarities))),
            "median_cosine_similarity": _safe_float(float(np.median(cosine_similarities))),
            "min_cosine_similarity": _safe_float(float(np.min(cosine_similarities))),
            "max_cosine_similarity": _safe_float(float(np.max(cosine_similarities))),
            "mean_delta": _safe_float(float(np.mean(deltas))),
        },
        "delta_threshold_counts": {
            f"{threshold:.2f}": int(np.sum(deltas > threshold)) for threshold in thresholds
        },
        "nearest_neighbor_preservation": nearest_neighbor,
        "cluster_impact": cluster_impact,
        "warnings": warnings,
    }
