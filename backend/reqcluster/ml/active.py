"""Active-learning helpers: uncertainty ranking by centroid margin. Pure."""

from __future__ import annotations

from typing import Any

import numpy as np

from reqcluster.errors import Invalid


def _unit(a):
    a = np.asarray(a, dtype=float)
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)


def uncertainty(probs, X, labels, top_k: int = 20) -> list[dict[str, Any]]:
    """Most uncertain first. Margin = cos(own centroid) - best other cosine (noise: nearest
    centroid counts as own, so margin is the gap between the two nearest). Small margin ranks first."""
    labels = np.asarray(labels)
    n = len(labels)
    if len(X) != n:
        raise Invalid("X and labels length differ")
    p = np.ones(n) if probs is None else np.asarray(probs, dtype=float)
    if len(p) != n:
        raise Invalid("probs and labels length differ")
    ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    margin = np.zeros(n)
    if ids and n:
        Xu = _unit(X)
        C = _unit(np.stack([Xu[labels == c].mean(axis=0) for c in ids]))
        sims = Xu @ C.T
        col = {c: j for j, c in enumerate(ids)}
        for i in range(n):
            s = sims[i]
            own = s.argmax() if labels[i] == -1 else col[int(labels[i])]
            rest = np.delete(s, own)
            margin[i] = s[own] - (rest.max() if len(rest) else -1.0)
    scored = [
        {
            "index": i,
            "cluster_id": int(labels[i]),
            "is_noise": bool(labels[i] == -1),
            "membership_prob": float(p[i]),
            "uncertainty": round(float(np.clip((1.0 - margin[i]) / 2.0, 0.0, 1.0)), 4),
        }
        for i in range(n)
    ]
    scored.sort(key=lambda d: d["uncertainty"], reverse=True)
    return scored[: max(0, int(top_k))]
