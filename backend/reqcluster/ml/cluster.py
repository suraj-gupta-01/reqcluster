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
