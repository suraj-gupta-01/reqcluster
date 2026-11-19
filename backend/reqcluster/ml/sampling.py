import numpy as np
from sklearn.metrics import silhouette_score


def silhouette(X, labels, sample_size: int = 2000, seed: int = 0) -> float | None:
    mask = labels != -1
    Xc, lc = np.asarray(X)[mask], labels[mask]
    if len(lc) < 3 or len(np.unique(lc)) < 2:
        return None
    n = len(lc)
    if n > sample_size:
        pick = np.random.default_rng(seed).choice(n, sample_size, replace=False)
        Xc, lc = Xc[pick], lc[pick]
        if len(np.unique(lc)) < 2:
            return None
    return float(silhouette_score(Xc, lc))


def knn(X: np.ndarray, k: int, block: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Exact cosine top-k per row (self excluded), blockwise."""
    X = np.asarray(X, dtype=np.float32)
    n = len(X)
    k = max(0, min(k, n - 1))
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    idx = np.zeros((n, k), dtype=np.int64)
    sim = np.zeros((n, k), dtype=np.float32)
    if k == 0:
        return idx, sim
    for s in range(0, n, block):
        S = Xn[s : s + block] @ Xn.T
        S[np.arange(S.shape[0]), np.arange(s, s + S.shape[0])] = -np.inf
        part = np.argpartition(-S, k - 1, axis=1)[:, :k]
        ps = np.take_along_axis(S, part, axis=1)
        o = np.argsort(-ps, axis=1, kind="stable")
        idx[s : s + block] = np.take_along_axis(part, o, axis=1)
        sim[s : s + block] = np.take_along_axis(ps, o, axis=1)
    return idx, sim
