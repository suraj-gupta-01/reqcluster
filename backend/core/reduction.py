import numpy as np
from typing import Tuple
import umap


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components_cluster: int = 10,
    n_components_viz: int = 2,
    metric: str = "cosine",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run UMAP dimensionality reduction.
    
    Returns:
        embeddings_10d: (N, 10) for clustering
        embeddings_2d:  (N, 2) for visualization
    """
    n_samples = embeddings.shape[0]

    # UMAP needs n_components < n_samples (and a few neighbours to work with).
    # Reject inputs that are too small to reduce meaningfully.
    if n_samples < 4:
        raise ValueError(
            f"Need at least 4 requirements to run dimensionality reduction (got {n_samples})."
        )

    # Adjust n_neighbors based on dataset size
    n_neighbors = min(15, max(2, n_samples - 1))

    # Clamp output dimensions so they always stay below n_samples
    cluster_dim = max(2, min(n_components_cluster, n_samples - 2))
    viz_dim = max(2, min(n_components_viz, n_samples - 2))

    # 10D reduction for clustering
    reducer_10d = umap.UMAP(
        n_components=cluster_dim,
        metric=metric,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        low_memory=False,
    )
    embeddings_10d = reducer_10d.fit_transform(embeddings)

    # 2D reduction for visualization
    reducer_2d = umap.UMAP(
        n_components=viz_dim,
        metric=metric,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        low_memory=False,
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    return embeddings_10d, embeddings_2d
