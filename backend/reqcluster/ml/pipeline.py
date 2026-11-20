import time

import numpy as np

from ..embeddings import calibration
from ..domain import ClusteringParams, ClusteringResult
from ..errors import Invalid
from . import cluster, labels as L, graph, keywords, reduce


def _cluster(x10, params: ClusteringParams, mcs: int, seed: int):
    hdb, probs = cluster.hdbscan(x10, mcs, params.min_samples)
    if params.clusterer != "ward" or (hdb == -1).all():
        return hdb, probs
    # ward draws the groups (ARI uav10k .963 -> .999, multidomain .688 -> .722), HDBSCAN keeps
    # flagging outliers (adversarial .413 vs .346 for pure ward): findings/exp/hybrid_noise.py
    w, _ = cluster.ward(x10, seed)
    noise = hdb == -1
    labels = L.relabel_contiguous(np.where(noise, -1, w))
    return labels, np.where(noise, 0.0, 1.0).astype(np.float32)


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
    labels, probs = step("cluster", 0.4, lambda: _cluster(x10, params, mcs, seed))
    clusters = step("keywords", 0.7, lambda: keywords.label_clusters(texts, labels))
    edges = step("graph", 0.85, lambda: graph.build(
        X, labels, params.sim_threshold if params.sim_threshold is not None
        else calibration().graph, params.top_k, seed, threads))
    if progress:
        progress(1.0, "done")

    if n < 5:
        warnings.append("fewer than 5 requirements: clustering skipped")
    if not clusters:
        warnings.append("all requirements were classified as noise")
    elif len(clusters) == 1:
        warnings.append("only one cluster found")
    return ClusteringResult(labels, probs, x10, xy, clusters, edges, warnings, durations)
