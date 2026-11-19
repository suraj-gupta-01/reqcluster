import numpy as np
from sklearn.cluster import HDBSCAN

from . import labels as L


def auto_min_cluster_size(n: int) -> int:
    return max(5, n // 50)


def hdbscan(x10: np.ndarray, min_cluster_size: int, min_samples: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(x10)
    if n < 5 or min_cluster_size > n:
        return np.full(n, -1, np.int64), np.zeros(n, np.float32)
    mcs = max(2, min_cluster_size)
    ms = min(max(1, min_samples), max(1, n - 1))
    model = HDBSCAN(min_cluster_size=mcs, min_samples=ms, cluster_selection_method="eom")
    raw = model.fit_predict(np.asarray(x10, dtype=np.float64))
    return L.relabel_contiguous(raw), np.asarray(model.probabilities_, dtype=np.float32)
