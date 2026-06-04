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
    
    # Adjust n_neighbors based on dataset size
    n_neighbors = min(15, max(2, n_samples - 1))

    # 10D reduction for clustering
    reducer_10d = umap.UMAP(
        n_components=n_components_cluster,
        metric=metric,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        low_memory=False,
    )
    embeddings_10d = reducer_10d.fit_transform(embeddings)

    # 2D reduction for visualization
    reducer_2d = umap.UMAP(
        n_components=n_components_viz,
        metric=metric,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        low_memory=False,
    )
    embeddings_2d = reducer_2d.fit_transform(embeddings)

    return embeddings_10d, embeddings_2d
