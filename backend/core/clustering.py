import numpy as np
from typing import Tuple, Optional
import hdbscan


def cluster_requirements(
    embeddings_10d: np.ndarray,
    min_cluster_size: Optional[int] = None,
    min_samples: int = 3,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run HDBSCAN clustering on 10D UMAP embeddings.
    
    Returns:
        labels: (N,) cluster assignments, -1 = noise
        probabilities: (N,) membership probabilities
    """
    n_samples = embeddings_10d.shape[0]

    if min_cluster_size is None:
        min_cluster_size = max(5, n_samples // 50)

    # Adjust min_samples for small datasets
    actual_min_samples = min(min_samples, min_cluster_size - 1)
    actual_min_samples = max(1, actual_min_samples)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=actual_min_samples,
        metric=metric,
        prediction_data=True,
        cluster_selection_method="eom",
    )

    clusterer.fit(embeddings_10d)
    labels = clusterer.labels_
    probabilities = clusterer.probabilities_

    return labels, probabilities
