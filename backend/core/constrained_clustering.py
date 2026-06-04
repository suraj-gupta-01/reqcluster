"""Constraint enforcement for clustering (Phase 5 active learning).

HDBSCAN has no native must-link / cannot-link support, so human constraints
captured in Phase 4 are enforced as a deterministic post-hoc repair layer over
an existing label assignment ("COP-HDBSCAN-lite"):

- must-link: connected components of must-linked points are collapsed onto a
  single representative cluster.
- cannot-link: when two cannot-linked points share a cluster, the
  lower-confidence point is moved to its nearest non-conflicting cluster (or to
  noise if none exists).

This is a repair layer, not a constrained optimiser; it is fast, deterministic,
and never fails. It operates purely on arrays and performs no DB or LLM access.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _find(parent: Dict[int, int], i: int) -> int:
    root = i
    while parent[root] != root:
        root = parent[root]
    while parent[i] != root:
        parent[i], i = root, parent[i]
    return root


def _union(parent: Dict[int, int], i: int, j: int) -> None:
    ri, rj = _find(parent, i), _find(parent, j)
    if ri != rj:
        parent[ri] = rj


def _cluster_centroids(
    embeddings: np.ndarray, labels: np.ndarray
) -> Dict[int, np.ndarray]:
    centroids: Dict[int, np.ndarray] = {}
    for cid in sorted(set(int(l) for l in labels if l != -1)):
        pts = embeddings[labels == cid]
        if pts.shape[0] == 0:
            continue
        c = pts.mean(axis=0)
        norm = np.linalg.norm(c)
        centroids[cid] = c / norm if norm > 0 else c
    return centroids


def apply_constraints(
    embeddings: np.ndarray,
    labels: np.ndarray,
    probabilities: Optional[np.ndarray],
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Enforce must-link / cannot-link constraints on a label assignment.

    Args:
        embeddings: (N, D) array used for centroid distances.
        labels: (N,) current cluster labels (-1 = noise).
        probabilities: (N,) membership confidences (or None).
        must_links / cannot_links: lists of (i, j) index pairs.

    Returns:
        (new_labels, info) where info reports how many points moved.
    """
    n = len(labels)
    labels = np.asarray(labels).astype(int).copy()
    if probabilities is None:
        probabilities = np.ones(n, dtype=float)
    else:
        probabilities = np.asarray(probabilities, dtype=float)

    moved_must = 0
    moved_cannot = 0

    # --- must-link: collapse connected components onto one cluster ---
    constrained = sorted({i for pair in must_links for i in pair if 0 <= i < n})
    if constrained:
        parent = {i: i for i in constrained}
        for i, j in must_links:
            if 0 <= i < n and 0 <= j < n:
                parent.setdefault(i, i)
                parent.setdefault(j, j)
                _union(parent, i, j)

        components: Dict[int, List[int]] = {}
        for i in parent:
            components.setdefault(_find(parent, i), []).append(i)

        next_new_label = (int(labels.max()) + 1) if labels.size else 0
        for members in components.values():
            member_labels = [labels[m] for m in members if labels[m] != -1]
            if member_labels:
                target = max(set(member_labels), key=member_labels.count)
            else:
                target = next_new_label
                next_new_label += 1
            for m in members:
                if labels[m] != target:
                    labels[m] = target
                    moved_must += 1

    # --- cannot-link: separate co-clustered conflicting points ---
    for i, j in cannot_links:
        if not (0 <= i < n and 0 <= j < n):
            continue
        if labels[i] == -1 or labels[i] != labels[j]:
            continue
        # Move the lower-confidence point off the shared cluster.
        loser = i if probabilities[i] <= probabilities[j] else j
        keeper_label = labels[i]
        centroids = _cluster_centroids(embeddings, labels)
        candidates = {
            cid: c for cid, c in centroids.items() if cid != keeper_label
        }
        new_label = -1
        if candidates:
            vec = embeddings[loser]
            vnorm = np.linalg.norm(vec)
            vec = vec / vnorm if vnorm > 0 else vec
            best_cid, best_sim = None, -np.inf
            for cid, c in candidates.items():
                sim = float(np.dot(vec, c))
                if sim > best_sim:
                    best_cid, best_sim = cid, sim
            new_label = best_cid if best_cid is not None else -1
        if labels[loser] != new_label:
            labels[loser] = new_label
            moved_cannot += 1

    info = {
        "must_link_pairs": len(must_links),
        "cannot_link_pairs": len(cannot_links),
        "points_moved_must_link": moved_must,
        "points_moved_cannot_link": moved_cannot,
    }
    return labels, info
