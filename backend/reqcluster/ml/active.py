"""Active-learning helpers: uncertainty ranking and clustering quality. Pure."""

from __future__ import annotations

from typing import Any

import numpy as np

from reqcluster.errors import Invalid
from reqcluster.ml import labels as L
from reqcluster.ml import sampling


def uncertainty(probs, X, labels, top_k: int = 20) -> list[dict[str, Any]]:
    """Most uncertain first: noise is 1.0, clustered points 1 - membership prob."""
    labels = np.asarray(labels)
    n = len(labels)
    if len(X) != n:
        raise Invalid("X and labels length differ")
    p = np.ones(n) if probs is None else np.asarray(probs, dtype=float)
    if len(p) != n:
        raise Invalid("probs and labels length differ")
    scored = []
    for i in range(n):
        noise = bool(labels[i] == -1)
        scored.append(
            {
                "index": i,
                "cluster_id": int(labels[i]),
                "is_noise": noise,
                "membership_prob": float(p[i]),
                "uncertainty": 1.0 if noise else round(max(0.0, 1.0 - float(p[i])), 4),
            }
        )
    scored.sort(key=lambda d: (d["uncertainty"], d["is_noise"]), reverse=True)
    return scored[: max(0, int(top_k))]


def quality(X10, labels, seed: int = 0) -> dict[str, Any]:
    labels = np.asarray(labels)
    n = len(labels)
    noise = int((labels == -1).sum())
    sil = sampling.silhouette(X10, labels, seed=seed)
    return {
        "n_clusters": len(L.cluster_ids(labels)),
        "noise_count": noise,
        "noise_rate": round(noise / n, 4) if n else 0.0,
        "silhouette": None if sil is None else round(sil, 4),
    }
