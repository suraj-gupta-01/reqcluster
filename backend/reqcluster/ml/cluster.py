import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import HDBSCAN

from . import labels as L
from .sampling import silhouette


def auto_min_cluster_size(n: int) -> int:
    return max(5, n // 50)


def hdbscan(x10: np.ndarray, min_cluster_size: int, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x10)
    if n < 5 or min_cluster_size > n:
        return np.full(n, -1, np.int64), np.zeros(n, np.float32)
    mcs = max(2, min_cluster_size)
    ms = min(max(1, min_samples), max(1, n - 1))
    # copy=True: never modify the caller's array in place (and the future sklearn default)
    model = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="eom", copy=True)
    raw = model.fit_predict(np.asarray(x10, dtype=np.float64))
    return L.relabel_contiguous(raw), np.asarray(model.probabilities_, dtype=np.float32)


def consensus(runs: list[np.ndarray], min_cluster_size: int, min_samples: int, X: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray]:
    """HDBSCAN on 1 - co-association (share of runs that put two rows in the same cluster), then
    `rescue` in the embedding space X.

    Probability = a row's mean co-association with the rest of its cluster: how often the runs
    agree on it. Noise 0.
    """
    n = len(runs[0])
    co = np.zeros((n, n))
    for lab in runs:
        co += (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
    co /= len(runs)
    mcs = max(2, min_cluster_size)
    ms = min(max(1, min_samples), max(1, n - 1))
    raw = HDBSCAN(min_cluster_size=mcs, min_samples=ms, metric="precomputed", copy=True,
                  cluster_selection_method="eom").fit_predict(1.0 - co)
    labels = rescue(X, L.relabel_contiguous(raw))
    return labels, _agreement(co, labels)


def _agreement(co: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(labels), np.float32)
    for idx in L.groups(labels).values():
        if len(idx) > 1:
            sub = co[np.ix_(idx, idx)]
            out[idx] = (sub.sum(axis=1) - sub.diagonal()) / (len(idx) - 1)
    return out


def rescue(X: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Noise rows join their nearest cluster (cosine to centroid) only if they are at least as
    close as that cluster's farthest member, i.e. inside its observed radius."""
    labels = labels.copy()
    noise = np.flatnonzero(labels == -1)
    ids, _ = L.centroids(X, labels)
    if not ids or not len(noise):
        return labels
    Xu = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    g = L.groups(labels)
    C = np.stack([Xu[g[c]].mean(axis=0) for c in ids])
    C /= np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    radius = np.array([(Xu[g[c]] @ C[i]).min() for i, c in enumerate(ids)])
    S = Xu[noise] @ C.T
    best = S.argmax(axis=1)
    ok = S[np.arange(len(noise)), best] >= radius[best]
    labels[noise[ok]] = np.asarray(ids)[best[ok]]
    return labels


def membership(x: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Centroid silhouette (b - a) / max(a, b) clipped to [0, 1]: a = distance to the row's own
    centroid, b = to the nearest other one. 0 for noise, 1 when there is no other cluster."""
    x = np.asarray(x, dtype=np.float64)
    out = np.zeros(len(labels), np.float32)
    ids, cents = L.centroids(x, labels)
    if not ids:
        return out
    if len(ids) == 1:
        out[labels != -1] = 1.0
        return out
    rows = np.flatnonzero(labels != -1)
    D = np.linalg.norm(x[rows, None, :] - cents[None, :, :], axis=2)
    own = np.searchsorted(ids, labels[rows])
    a = D[np.arange(len(rows)), own]
    D[np.arange(len(rows)), own] = np.inf
    b = D.min(axis=1)
    out[rows] = np.clip((b - a) / np.maximum(np.maximum(a, b), 1e-12), 0.0, 1.0)
    return out


def ward(x10: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Ward linkage, k by sampled silhouette over [8, min(40, n // 5)]; never any noise."""
    n = len(x10)
    if n < 5:
        return np.full(n, -1, np.int64), np.zeros(n, np.float32)
    x = np.asarray(x10, dtype=np.float64)
    hi = max(2, min(40, n // 5))
    lo = min(8, hi)
    Z = linkage(x, "ward")
    best, best_s = None, -2.0
    for k in range(lo, hi + 1):
        lab = fcluster(Z, k, "maxclust") - 1
        s = silhouette(x, lab, seed=seed)
        if s is not None and s > best_s:
            best, best_s = lab, s
    if best is None:
        best = np.zeros(n, np.int64)
    labels = L.relabel_contiguous(best.astype(np.int64))
    return labels, membership(x, labels)
