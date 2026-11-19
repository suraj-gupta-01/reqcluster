import time

import numpy as np

from ..domain import ClusteringParams, ClusteringResult
from ..errors import Invalid
from . import cluster, graph, keywords, reduce


def run(embeddings, texts, params: ClusteringParams, *, seed: int, threads: int = 1,
        progress=None) -> ClusteringResult:
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2 or len(X) == 0:
        raise Invalid("embeddings must be a non-empty 2-D array")
    n = len(X)
    if len(texts) != n:
        raise Invalid(f"embeddings ({n}) and texts ({len(texts)}) length mismatch")

    durations: dict[str, float] = {}
    warnings: list[str] = []

    def step(name, frac, fn):
        if progress:
            progress(frac, name)
        t = time.perf_counter()
        out = fn()
        durations[name] = time.perf_counter() - t
        return out

    x10, xy = step("reduce", 0.0, lambda: reduce.reduce(X, seed))
    mcs = params.min_cluster_size or cluster.auto_min_cluster_size(n)
    labels, probs = step("cluster", 0.4, lambda: cluster.hdbscan(x10, mcs, params.min_samples))
    clusters = step("keywords", 0.7, lambda: keywords.label_clusters(texts, labels))
    edges = step("graph", 0.85, lambda: graph.build(
        X, labels, params.sim_threshold, params.top_k, seed, threads))
    if progress:
        progress(1.0, "done")

    if n < 5:
        warnings.append("fewer than 5 requirements: clustering skipped")
    if not clusters:
        warnings.append("all requirements were classified as noise")
    elif len(clusters) == 1:
        warnings.append("only one cluster found")
    return ClusteringResult(labels, probs, x10, xy, clusters, edges, warnings, durations)
