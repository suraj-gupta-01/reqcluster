import numpy as np

from . import labels as L
from . import sampling


def compute(labels: np.ndarray, x10: np.ndarray, seed: int) -> dict:
    n = len(labels)
    noise = int((labels == -1).sum())
    rate = noise / n if n else 0.0
    return {
        "n_clusters": len(L.cluster_ids(labels)),
        "noise_count": noise,
        "noise_rate": rate,
        "coverage": 1 - rate,
        "silhouette": sampling.silhouette(x10, labels, seed=seed),
    }
