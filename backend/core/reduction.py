import numpy as np
from typing import Tuple, Any
import umap

try:
    import torch
    import cuml
    GPU_AVAILABLE = torch.cuda.is_available()
except Exception:
    # Not just ImportError: a CPU-only torch whose native DLLs fail to load
    # raises OSError (e.g. Windows WinError 1114). That must also fall back to
    # the CPU path instead of crashing the whole import.
    GPU_AVAILABLE = False


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components_cluster: int = 10,
    n_components_viz: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
    return_reducers: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Any, Any]:
    """
    Run UMAP dimensionality reduction.
    
    Returns:
        embeddings_10d: (N, 10) for clustering
        embeddings_2d:  (N, 2) for visualization
        (Optionally) reducer_10d, reducer_2d if return_reducers=True
    """
    n_samples = embeddings.shape[0]

    # UMAP needs n_components < n_samples (and a few neighbours to work with).
    # Reject inputs that are too small to reduce meaningfully.
    if n_samples < 4:
        raise ValueError(
            f"Need at least 4 requirements to run dimensionality reduction (got {n_samples})."
        )

    # Scale n_neighbors with dataset size. A fixed 15 is too local for large sets
    # (UMAP over-fragments -> HDBSCAN marks many boundary points as noise); a wider
    # neighborhood captures global structure and keeps big runs clean. Small sets
    # stay at 15 for tight local detail.
    if n_samples <= 15:
        n_neighbors = max(2, n_samples - 1)
    else:
        n_neighbors = int(min(80, max(15, n_samples // 500)))

    # Clamp output dimensions so they always stay below n_samples
    cluster_dim = max(2, min(n_components_cluster, n_samples - 2))
    viz_dim = max(2, min(n_components_viz, n_samples - 2))

    # Set adaptive parameters
    if n_samples < 4000:
        active_random_state = random_state
        n_jobs = 1
        low_memory = False
    else:
        active_random_state = None
        n_jobs = -1
        low_memory = True

    # PCA pre-reduction to denoise and speed up kNN for larger datasets
    if n_samples > 50:
        from sklearn.decomposition import PCA
        pca_dim = min(50, n_samples - 2)
        pca = PCA(n_components=pca_dim, random_state=random_state)
        embeddings_reduced = pca.fit_transform(embeddings)
    else:
        embeddings_reduced = embeddings

    # ── GPU Path ───────────────────────────────────────────────────────────
    if GPU_AVAILABLE:
        try:
            reducer_10d = cuml.manifold.UMAP(
                n_components=cluster_dim,
                n_neighbors=n_neighbors,
                min_dist=0.0,
                random_state=active_random_state,
            )
            embeddings_10d = reducer_10d.fit_transform(embeddings_reduced)

            reducer_2d = cuml.manifold.UMAP(
                n_components=viz_dim,
                n_neighbors=n_neighbors,
                min_dist=0.1,
                random_state=active_random_state,
            )
            embeddings_2d = reducer_2d.fit_transform(embeddings_reduced)
            if return_reducers:
                return embeddings_10d, embeddings_2d, reducer_10d, reducer_2d
            return embeddings_10d, embeddings_2d
        except Exception:
            # Fall back to CPU path if GPU execution fails
            pass

    # ── CPU Path ───────────────────────────────────────────────────────────
    # 10D reduction for clustering
    reducer_10d = umap.UMAP(
        n_components=cluster_dim,
        metric=metric,
        random_state=active_random_state,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        low_memory=low_memory,
        n_jobs=n_jobs,
        init="random",
    )
    embeddings_10d = reducer_10d.fit_transform(embeddings_reduced)

    # Extract nearest neighbors calculated by 10D fit to reuse
    precomputed_knn = None
    if (
        hasattr(reducer_10d, "_knn_indices")
        and reducer_10d._knn_indices is not None
        and hasattr(reducer_10d, "_knn_dists")
        and reducer_10d._knn_dists is not None
    ):
        precomputed_knn = (reducer_10d._knn_indices, reducer_10d._knn_dists)

    # 2D reduction for visualization
    umap_kwargs = {
        "n_components": viz_dim,
        "metric": metric,
        "random_state": active_random_state,
        "n_neighbors": n_neighbors,
        "min_dist": 0.1,
        "low_memory": low_memory,
        "n_jobs": n_jobs,
        "init": "random",
    }
    if precomputed_knn is not None:
        umap_kwargs["precomputed_knn"] = precomputed_knn

    reducer_2d = umap.UMAP(**umap_kwargs)
    embeddings_2d = reducer_2d.fit_transform(embeddings_reduced)

    if return_reducers:
        return embeddings_10d, embeddings_2d, reducer_10d, reducer_2d
    return embeddings_10d, embeddings_2d

