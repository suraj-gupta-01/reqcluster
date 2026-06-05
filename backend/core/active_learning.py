"""Active-learning utilities: uncertainty sampling and clustering quality.

Selects the requirements whose cluster assignment is least certain (the most
valuable to send for human review) and computes quality metrics used to track
improvement across constrained re-clustering iterations.

Pure functions; no DB or LLM access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from sklearn.metrics import silhouette_score
except Exception:  # pragma: no cover - sklearn always present in this project
    silhouette_score = None


def uncertainty_queue(
    labels: np.ndarray,
    probabilities: Optional[np.ndarray],
    texts: List[str],
    req_ids: List[str],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """Rank requirements by clustering uncertainty (most uncertain first).

    Noise points are maximally uncertain; clustered points are scored by
    ``1 - membership_probability``.
    """
    n = len(labels)
    labels = np.asarray(labels)
    if probabilities is None:
        probabilities = np.ones(n, dtype=float)
    else:
        probabilities = np.asarray(probabilities, dtype=float)

    scored = []
    for i in range(n):
        is_noise = bool(labels[i] == -1)
        prob = float(probabilities[i]) if i < len(probabilities) else 0.0
        uncertainty = 1.0 if is_noise else max(0.0, 1.0 - prob)
        scored.append(
            {
                "index": i,
                "req_id": req_ids[i] if i < len(req_ids) else f"REQ-{i + 1:03d}",
                "text": texts[i] if i < len(texts) else "",
                "cluster_id": int(labels[i]),
                "is_noise": is_noise,
                "membership_prob": prob,
                "uncertainty": round(uncertainty, 4),
            }
        )

    scored.sort(key=lambda d: (d["uncertainty"], d["is_noise"]), reverse=True)
    return scored[: max(0, int(top_k))]


def clustering_quality(
    embeddings: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Silhouette (non-noise), noise rate, and cluster count for a labelling."""
    labels = np.asarray(labels)
    n = len(labels)
    noise_count = int((labels == -1).sum())
    noise_rate = round(noise_count / n, 4) if n else 0.0
    non_noise = labels != -1
    n_clusters = len(set(int(l) for l in labels[non_noise]))

    silhouette: Optional[float] = None
    if silhouette_score is not None and n_clusters >= 2:
        idx = np.where(non_noise)[0]
        sub_labels = labels[idx]
        # silhouette needs >= 2 labels and fewer clusters than samples.
        if len(idx) > n_clusters and len(set(int(l) for l in sub_labels)) >= 2:
            try:
                silhouette = round(float(silhouette_score(embeddings[idx], sub_labels)), 4)
            except Exception:
                silhouette = None

    return {
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "noise_rate": noise_rate,
        "silhouette": silhouette,
    }
