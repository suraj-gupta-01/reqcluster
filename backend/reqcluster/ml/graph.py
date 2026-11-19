import numpy as np

from . import sampling


def _hnsw_edges(Xn: np.ndarray, k: int, seed: int, threads: int):
    try:
        import hnswlib
    except ImportError:
        return None
    m, dim = Xn.shape
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=m, ef_construction=200, M=16, random_seed=seed)
    index.set_num_threads(threads)
    index.add_items(Xn, np.arange(m))
    index.set_ef(max(k + 1, 64))
    nb, dist = index.knn_query(Xn, k=min(k + 1, m))
    return nb.astype(np.int64), (1.0 - dist).astype(np.float32)


def build(embeddings, labels, threshold: float, k: int = 10, seed: int = 0,
          threads: int = 1, max_edges: int = 4000) -> list[tuple[int, int, float]]:
    nodes = np.flatnonzero(labels != -1)
    m = len(nodes)
    if m < 2 or k < 1:
        return []
    X = np.asarray(embeddings, dtype=np.float32)[nodes]
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    Xn = X / norms
    res = _hnsw_edges(Xn, k, seed, threads)
    nb, sim = res if res is not None else sampling.knn(Xn, k)
    best: dict[tuple[int, int], float] = {}
    per_node: dict[int, list[tuple[float, tuple[int, int]]]] = {}
    for li in range(m):
        for lj, s in zip(nb[li], sim[li], strict=True):
            lj = int(lj)
            if lj == li or s < threshold:
                continue
            key = tuple(sorted((int(nodes[li]), int(nodes[lj]))))
            best[key] = max(best.get(key, -1.0), float(s))
            per_node.setdefault(li, []).append((float(s), key))
    order = lambda k: (-best[k], k[0], k[1])  # noqa: E731
    # Reserve each node's 2 best edges so tight duplicate groups cannot eat the
    # whole budget and leave looser clusters isolated.
    keep = {k for c in per_node.values() for _, k in sorted(c, key=lambda t: (-t[0], t[1]))[:2]}
    chosen = sorted(keep, key=order)[:max_edges]
    chosen += [k for k in sorted(best.keys() - keep, key=order)][: max_edges - len(chosen)]
    chosen.sort(key=order)
    return [(a, b, round(best[(a, b)], 4)) for a, b in chosen]
