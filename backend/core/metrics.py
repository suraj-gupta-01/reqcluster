"""Clustering validation metrics.

Two families:

- **Intrinsic** (always available): how well-separated the clusters are, from the
  geometry alone - silhouette on the 2-D layout, noise rate, cluster count.
- **Extrinsic** (only when the upload carries ground-truth group labels in the
  `module` column): how well the clusters match the known grouping - Adjusted Rand
  Index, Normalized Mutual Information, homogeneity / completeness / V-measure,
  cluster purity (surfaced as "accuracy"), Hungarian-matched accuracy, and a
  confusion matrix.

For a labelled dataset (e.g. the UAV-FMS sets, where `module` is the true
subsystem) the extrinsic block is the real "accuracy". For unlabelled user data
only the intrinsic block is returned.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _has_ground_truth(gt: List[str], n: int) -> bool:
    if not gt or len(gt) != n:
        return False
    distinct = {g for g in gt if g and g.strip() and g.strip().lower() != "general"}
    return len(distinct) >= 2


def compute_metrics(
    cluster_labels: Sequence[int],
    ground_truth: Optional[Sequence[str]] = None,
    coords: Optional[Sequence[Sequence[float]]] = None,
) -> Dict[str, Any]:
    """Compute intrinsic + (if labelled) extrinsic clustering metrics."""
    labels = np.asarray([int(x) if x is not None else -1 for x in cluster_labels])
    n = int(labels.shape[0])
    noise = int((labels == -1).sum())
    n_clusters = len({int(x) for x in labels if x != -1})

    out: Dict[str, Any] = {
        "n": n,
        "n_clusters": n_clusters,
        "noise": noise,
        "noise_pct": round(100.0 * noise / n, 2) if n else 0.0,
        "coverage_pct": round(100.0 * (n - noise) / n, 2) if n else 0.0,
        "has_ground_truth": False,
    }

    non_noise = labels != -1

    # --- intrinsic: silhouette on the 2-D layout (non-noise, >= 2 clusters) ---
    if coords is not None and non_noise.sum() >= 3:
        try:
            from sklearn.metrics import silhouette_score

            X = np.asarray(coords, dtype=np.float64)[non_noise]
            y = labels[non_noise]
            if len(set(y.tolist())) >= 2:
                out["silhouette"] = round(float(silhouette_score(X, y)), 4)
        except Exception:
            pass

    # --- extrinsic vs ground truth ---
    gt = [str(g or "").strip() for g in (ground_truth or [])]
    if _has_ground_truth(gt, n) and non_noise.sum() >= 2:
        from sklearn.metrics import (
            adjusted_rand_score,
            homogeneity_completeness_v_measure,
            normalized_mutual_info_score,
        )

        cl = labels[non_noise]
        tr = np.asarray(gt)[non_noise]

        homogeneity, completeness, v_measure = homogeneity_completeness_v_measure(tr, cl)

        # cluster purity: each cluster's most common true group / clustered points
        by_cluster: Dict[int, Counter] = {}
        for cid, t in zip(cl.tolist(), tr.tolist()):
            by_cluster.setdefault(int(cid), Counter())[t] += 1
        purity = sum(c.most_common(1)[0][1] for c in by_cluster.values()) / len(cl)

        # Hungarian-matched accuracy: optimal cluster→group 1-to-1 assignment
        hung_acc = _hungarian_accuracy(cl, tr)

        # Confusion matrix: rows=clusters (sorted), cols=true groups (sorted alpha)
        conf_matrix, cluster_ids, group_names = _confusion_matrix(cl, tr)

        out.update({
            "has_ground_truth": True,
            "n_true_groups": int(len(set(tr.tolist()))),
            "ari": round(float(adjusted_rand_score(tr, cl)), 4),
            "nmi": round(float(normalized_mutual_info_score(tr, cl)), 4),
            "homogeneity": round(float(homogeneity), 4),
            "completeness": round(float(completeness), 4),
            "v_measure": round(float(v_measure), 4),
            "purity_pct": round(100.0 * purity, 2),
            "accuracy_pct": round(100.0 * purity, 2),  # purity alias, kept for compat
            "hungarian_accuracy_pct": round(100.0 * hung_acc, 2),
            "confusion_matrix_rows": conf_matrix,
            "confusion_matrix_cluster_ids": cluster_ids,
            "confusion_matrix_group_names": group_names,
        })

    return out


def _hungarian_accuracy(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Compute optimal cluster→group matched accuracy via Hungarian algorithm.

    Each predicted cluster is assigned to exactly one true group (1-to-1).
    The total correctly assigned points divided by total clustered points is
    the Hungarian accuracy — stricter than purity which allows many-to-one.
    Returns 0.0 if fewer than 2 clusters or groups exist.
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        return 0.0

    cluster_ids = sorted(set(int(c) for c in cluster_labels))
    group_ids = sorted(set(str(t) for t in true_labels))
    if len(cluster_ids) < 2 or len(group_ids) < 1:
        return 0.0

    cid_index = {c: i for i, c in enumerate(cluster_ids)}
    gid_index = {g: j for j, g in enumerate(group_ids)}

    # Build cost matrix (negated counts so linear_sum_assignment minimises)
    cost = np.zeros((len(cluster_ids), len(group_ids)), dtype=np.int64)
    for c, t in zip(cluster_labels.tolist(), true_labels.tolist()):
        cost[cid_index[int(c)], gid_index[str(t)]] += 1

    row_ind, col_ind = linear_sum_assignment(-cost)
    matched = int(cost[row_ind, col_ind].sum())
    return matched / max(len(cluster_labels), 1)


def _confusion_matrix(
    cluster_labels: np.ndarray,
    true_labels: np.ndarray,
) -> tuple[list[list[int]], list[int], list[str]]:
    """Return confusion matrix as list-of-lists (cluster rows × true-group columns).

    Returns:
        matrix: list[list[int]] — counts, rows = sorted cluster ids, cols = sorted group names
        cluster_ids: list[int] — row labels
        group_names: list[str] — column labels
    """
    cluster_ids = sorted(set(int(c) for c in cluster_labels))
    group_names = sorted(set(str(t) for t in true_labels))

    cid_index = {c: i for i, c in enumerate(cluster_ids)}
    gid_index = {g: j for j, g in enumerate(group_names)}

    matrix = [[0] * len(group_names) for _ in cluster_ids]
    for c, t in zip(cluster_labels.tolist(), true_labels.tolist()):
        matrix[cid_index[int(c)]][gid_index[str(t)]] += 1

    return matrix, cluster_ids, group_names
