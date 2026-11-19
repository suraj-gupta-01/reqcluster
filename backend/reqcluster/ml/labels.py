import numpy as np


def normalize(labels) -> np.ndarray:
    out = [
        -1 if v is None or (isinstance(v, float) and np.isnan(v)) else int(v)
        for v in labels
    ]
    return np.asarray(out, dtype=np.int64)


def clustered_mask(labels: np.ndarray) -> np.ndarray:
    return labels != -1


def cluster_ids(labels: np.ndarray) -> list[int]:
    return [int(c) for c in np.unique(labels) if c != -1]


def groups(labels: np.ndarray) -> dict[int, np.ndarray]:
    labels = np.asarray(labels)
    order = np.argsort(labels, kind="stable")
    uniq, starts = np.unique(labels[order], return_index=True)
    parts = np.split(order, starts[1:])
    return {int(u): p for u, p in zip(uniq, parts, strict=True) if u != -1}


def centroids(X: np.ndarray, labels: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = groups(labels)
    ids = sorted(g)
    if not ids:
        return ids, np.zeros((0, X.shape[1]), dtype=np.float32)
    return ids, np.stack([X[g[c]].mean(axis=0) for c in ids])


def relabel_contiguous(labels: np.ndarray) -> np.ndarray:
    out = np.full(len(labels), -1, dtype=np.int64)
    mapping: dict[int, int] = {}
    for i, v in enumerate(labels):
        if v == -1:
            continue
        out[i] = mapping.setdefault(int(v), len(mapping))
    return out
