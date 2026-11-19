import threading

import numpy as np
from sklearn.decomposition import PCA


# numba's default "workqueue" threading layer aborts the whole process when two threads run
# parallel kernels at once (startup warm-up + a job, or job_workers > 1). Serialize UMAP.
_UMAP_LOCK = threading.Lock()


def reduce(X: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float32)
    n, d = X.shape
    if n < 20:
        if n == 1:
            return np.zeros((1, 1), np.float32), np.zeros((1, 2), np.float32)
        P = PCA(n_components=min(10, n - 1, d), svd_solver="full").fit_transform(X)
        xy = np.zeros((n, 2), np.float32)
        c = min(2, P.shape[1])
        xy[:, :c] = P[:, :c]
        return P.astype(np.float32), xy

    with _UMAP_LOCK:
        return _umap(X, seed, n)


def _umap(X: np.ndarray, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    import umap  # lazy: slow import

    # No PCA50 before UMAP: on raw cosine embeddings ARI rises uav2k .934->.987, uav10k .908->.958,
    # multidomain .636->.649, and 10k is faster (findings/exp/pca_threshold.py, 3 seeds).
    nn = min(80, max(15, n // 100), n - 1)
    x10 = umap.UMAP(
        n_components=10, metric="cosine", n_neighbors=nn, min_dist=0.0, n_epochs=200,
        init="random", random_state=seed, compatibility_layout=True,
    ).fit_transform(X)
    # viz only: 100 epochs at 10k rows 28.9s -> 5.4s (vs 200); clustering uses x10
    xy = umap.UMAP(
        n_components=2, metric="euclidean", n_neighbors=min(15, n - 1), min_dist=0.1,
        n_epochs=100, init="random", random_state=seed, compatibility_layout=True,
    ).fit_transform(x10)
    return x10.astype(np.float32), xy.astype(np.float32)
