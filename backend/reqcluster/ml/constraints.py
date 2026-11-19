"""Must-link / cannot-link repair over an existing labelling. Pure, deterministic."""

from __future__ import annotations

from collections import Counter

import numpy as np

from reqcluster.errors import Invalid
from reqcluster.ml import labels as L


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1)


def _find(parent: dict[int, int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _check(pairs, n: int) -> list[tuple[int, int]]:
    out = []
    for a, b in pairs:
        if not (0 <= a < n and 0 <= b < n):
            raise Invalid(f"constraint pair ({a}, {b}) out of range")
        if a != b:
            out.append((int(a), int(b)))
    return out


def apply(X, labels, must, cannot):
    """Return (new_labels, moved). Must-link components are never split."""
    X = np.asarray(X, dtype=np.float64)
    labels = L.normalize(labels)
    n = len(labels)
    if len(X) != n:
        raise Invalid("X and labels length differ")
    must, cannot = _check(must, n), _check(cannot, n)
    orig = labels.copy()
    out = labels.copy()

    parent = {i: i for i in range(n)}
    for a, b in must:
        parent[_find(parent, a)] = _find(parent, b)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(_find(parent, i), []).append(i)
    for a, b in cannot:
        if _find(parent, a) == _find(parent, b):
            raise Invalid(f"pair ({min(a, b)}, {max(a, b)}) is both must-link and cannot-link")

    ids, cents = L.centroids(X, orig)
    cents = _unit(cents)

    def ranked(members: list[int]) -> list[int]:
        """Cluster ids by similarity to the component mean, best first."""
        sims = cents @ _unit(X[members].mean(axis=0))
        return [ids[j] for j in np.argsort(-sims, kind="stable")]

    next_new = int(orig.max()) + 1 if n and orig.max() >= 0 else 0
    for members in (m for m in comps.values() if len(m) > 1):
        ls = [int(orig[m]) for m in members if orig[m] != -1]
        if ls:
            c = Counter(ls)
            top = max(c.values())
            target = min(k for k, v in c.items() if v == top)
        elif ids:
            target = ranked(members)[0]
        else:
            target, next_new = next_new, next_new + 1
        out[members] = target

    partners: dict[int, list[int]] = {}
    for a, b in cannot:
        partners.setdefault(a, []).append(b)
        partners.setdefault(b, []).append(a)

    for a, b in cannot:
        if out[a] == -1 or out[a] != out[b]:
            continue
        ca, cb = comps[_find(parent, a)], comps[_find(parent, b)]
        mover = ca if len(ca) <= len(cb) else cb
        banned = {int(out[p]) for m in mover for p in partners.get(m, [])}
        banned.add(int(out[a]))
        new = next((c for c in ranked(mover) if c not in banned), -1) if ids else -1
        out[mover] = new

    return out, int((out != orig).sum())


def from_correction(labels, index: int, new_cluster_id: int, old_cluster_id=None, *, X, top_n: int = 3):
    """Constraints for one human move of `index` into `new_cluster_id`.

    `labels` already reflect the move. Must-link to the top_n members of the new
    cluster nearest its centroid; cannot-link to the same for the old cluster.
    """
    labels = L.normalize(labels).copy()
    X = _unit(np.asarray(X, dtype=np.float64))
    if not 0 <= index < len(labels) or len(X) != len(labels):
        raise Invalid("index out of range or X/labels length differ")

    def reps(cid: int) -> list[int]:
        idx = np.where(labels == cid)[0]
        if len(idx) == 0:
            return []
        c = _unit(X[idx].mean(axis=0))
        order = np.argsort(-(X[idx] @ c), kind="stable")[:top_n]
        return [int(idx[o]) for o in order if idx[o] != index]

    must: set[tuple[int, int]] = set()
    cannot: set[tuple[int, int]] = set()
    if new_cluster_id is not None and new_cluster_id != -1:
        must = {(min(index, r), max(index, r)) for r in reps(new_cluster_id)}
    if old_cluster_id is not None and old_cluster_id != -1:
        labels[index] = old_cluster_id  # old reps are computed as before the move
        cannot = {(min(index, r), max(index, r)) for r in reps(old_cluster_id)}
    return sorted(must), sorted(cannot)
