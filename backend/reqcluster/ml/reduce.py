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


# Exact cosine kNN via blocked BLAS matmul (multi-threaded) instead of UMAP's single-threaded
# numba pairwise pass (<4096 rows) / NN-descent. Faster up to the 50k upload cap: 35k rows
# reduce 220 s -> 118 s (kNN itself 18.6 s). Blocks shrink above 20k to cap the matmul buffer.
EXACT_KNN_MAX = 50_000
# Below this UMAP's own (also exact) pairwise path is as fast and steadier on tiny sets
# (aerospace 64 rows, 5 seeds: ARI .488+-.01 own vs .442+-.046 BLAS; equal on 100/371 rows).
EXACT_KNN_MIN = 1_000


def _exact_knn(X: np.ndarray, nn: int):
    from . import sampling

    block = 2048 if len(X) <= 20_000 else 1024  # 1024 x 50k float32 = 205 MB per block
    idx, sim = sampling.knn(X, nn - 1, block=block)  # excludes self
    n = len(X)
    knn_i = np.hstack([np.arange(n)[:, None], idx]).astype(np.int32)
    knn_d = np.hstack([np.zeros((n, 1), np.float32), np.clip(1.0 - sim, 0.0, 2.0)])
    return knn_i, knn_d.astype(np.float32), None


def _viz_epochs(n: int) -> int:
    # viz only (clustering uses x10). 35k rows, 3 seeds (findings/exp/umap_scale.py): 100 epochs
    # 30.6 s -> 50 epochs 16.2 s (30 ep 10.8 s; reusing the 10-D kNN 12.9 s, no better than plain).
    # 10-D options all rejected at 35k: n_epochs 100 ARI .948 -> .830, 140 -> .917, Adam 200 -> .368.
    return 100 if n <= 20_000 else 50


def _umap(X: np.ndarray, seed: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    import umap  # lazy: slow import

    # No PCA50 before UMAP: on raw cosine embeddings ARI rises uav2k .934->.987, uav10k .908->.958,
    # multidomain .636->.649, and 10k is faster (findings/exp/pca_threshold.py, 3 seeds).
    nn = min(80, max(15, n // 100), n - 1)
    x10 = umap.UMAP(
        n_components=10, metric="cosine", n_neighbors=nn, min_dist=0.0, n_epochs=200,
        init="random", random_state=seed, compatibility_layout=True,
        precomputed_knn=(_exact_knn(X, nn) if EXACT_KNN_MIN <= n <= EXACT_KNN_MAX
                         else (None, None, None)),
    ).fit_transform(X)
    # viz only: 100 epochs at 10k rows 28.9s -> 5.4s (vs 200); clustering uses x10
    xy = umap.UMAP(
        n_components=2, metric="euclidean", n_neighbors=min(15, n - 1), min_dist=0.1,
        n_epochs=_viz_epochs(n), init="random", random_state=seed, compatibility_layout=True,
    ).fit_transform(x10)
    return x10.astype(np.float32), xy.astype(np.float32)
