"""Embedding comparison and base-vs-enriched ablation. Pure: embed_fn is injected."""
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from ..domain import ClusteringParams
from ..errors import Invalid
from ..texts import build_texts
from . import labels as L
from . import pipeline, sampling

DEFAULT_DELTA_THRESHOLDS = (0.05, 0.10, 0.20)
DEFAULT_NEIGHBOR_K = 5
MAX_NEIGHBOR_ITEMS = 2_000


def elapsed_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000.0, 3)


def _f(v) -> float | None:
    if v is None or not np.isfinite(v):
        return None
    return round(float(v), 6)


@dataclass
class Aggregate:
    mean_cosine_similarity: float | None = None
    median_cosine_similarity: float | None = None
    min_cosine_similarity: float | None = None
    max_cosine_similarity: float | None = None
    mean_delta: float | None = None


@dataclass
class NeighborPreservation:
    k: int = 0
    score: float | None = None
    evaluated_count: int = 0
    max_items: int = MAX_NEIGHBOR_ITEMS
    truncated: bool = False


@dataclass
class ClusterImpact:
    adjusted_rand_index: float | None = None
    normalized_mutual_info: float | None = None
    base_silhouette_score_10d: float | None = None
    candidate_silhouette_score_10d: float | None = None
    silhouette_score_delta: float | None = None


@dataclass
class RequirementDelta:
    index: int
    cosine_similarity: float | None
    cosine_distance_delta: float | None


@dataclass
class EmbeddingComparison:
    n_requirements: int = 0
    per_requirement: list[RequirementDelta] = field(default_factory=list)
    aggregate: Aggregate = field(default_factory=Aggregate)
    delta_threshold_counts: dict[str, int] = field(default_factory=dict)
    nearest_neighbor_preservation: NeighborPreservation = field(default_factory=NeighborPreservation)
    cluster_impact: ClusterImpact | None = None
    warnings: list[str] = field(default_factory=list)


def _empty(warnings: list[str], thresholds) -> EmbeddingComparison:
    return EmbeddingComparison(
        delta_threshold_counts={f"{t:.2f}": 0 for t in thresholds}, warnings=warnings)


def _as_matrix(value, name: str, warnings: list[str]) -> np.ndarray | None:
    try:
        arr = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        warnings.append(f"{name} could not be converted to a numeric matrix.")
        return None
    if arr.ndim != 2:
        warnings.append(f"{name} must be a 2D matrix.")
        return None
    if arr.shape[1] == 0:
        warnings.append(f"{name} must have at least one embedding dimension.")
        return None
    if not np.all(np.isfinite(arr)):
        warnings.append(f"{name} contained NaN or infinite values; invalid values were replaced.")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr


def _row_normalize(arr: np.ndarray, name: str, warnings: list[str]) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    zero = norms[:, 0] <= 1e-12
    if np.any(zero):
        warnings.append(f"{name} contained zero-norm rows; cosine values for them may be 0.")
    norms[zero] = 1.0
    return arr / norms


def _neighbor_preservation(base: np.ndarray, cand: np.ndarray, k: int, max_items: int,
                           warnings: list[str]) -> NeighborPreservation:
    n = base.shape[0]
    if n <= 1:
        warnings.append("Nearest-neighbor preservation requires at least two requirements.")
        return NeighborPreservation(k=0, score=None, evaluated_count=n, max_items=max_items)
    m = min(n, max_items)
    truncated = m < n
    if truncated:
        warnings.append(f"Nearest-neighbor preservation evaluated the first {m} requirements.")
    k = min(max(1, int(k)), m - 1)
    ib, _ = sampling.knn(base[:m], k)
    ic, _ = sampling.knn(cand[:m], k)
    overlaps = []
    for a, b in zip(ib, ic, strict=True):
        sa, sb = set(a.tolist()), set(b.tolist())
        union = sa | sb
        overlaps.append(1.0 if not union else len(sa & sb) / len(union))
    return NeighborPreservation(k=k, score=_f(np.mean(overlaps)), evaluated_count=m,
                                max_items=max_items, truncated=truncated)


def _labels_array(labels, rows: int, name: str, warnings: list[str]) -> np.ndarray | None:
    if labels is None:
        return None
    arr = np.asarray(labels)
    if arr.ndim != 1 or arr.shape[0] != rows:
        warnings.append(f"{name} must be a 1D label array matching the embeddings row count.")
        return None
    return arr.astype(np.int64, copy=False)


def _silhouette(x, labels: np.ndarray | None, name: str, warnings: list[str],
                seed: int) -> float | None:
    if x is None or labels is None:
        return None
    if x.ndim != 2 or x.shape[0] != labels.shape[0]:
        warnings.append(f"{name} silhouette skipped because embeddings and labels do not align.")
        return None
    mask = L.clustered_mask(labels)
    if int(mask.sum()) < 3:
        warnings.append(f"{name} silhouette skipped because too few non-noise points exist.")
        return None
    k = len(L.cluster_ids(labels))
    if k < 2:
        warnings.append(f"{name} silhouette skipped because fewer than two clusters exist.")
        return None
    if k >= int(mask.sum()):
        warnings.append(f"{name} silhouette skipped because every point is its own cluster.")
        return None
    return sampling.silhouette(x, labels, seed=seed)


def _cluster_impact(lb, lc, x_base, x_cand, rows: int, warnings: list[str],
                    seed: int) -> ClusterImpact | None:
    if lb is None and lc is None:
        return None
    res = ClusterImpact()
    base = _labels_array(lb, rows, "labels_base", warnings)
    cand = _labels_array(lc, rows, "labels_candidate", warnings)
    if base is None or cand is None:
        return res
    res.adjusted_rand_index = _f(adjusted_rand_score(base, cand))
    res.normalized_mutual_info = _f(normalized_mutual_info_score(base, cand))
    sb = _silhouette(x_base, base, "base", warnings, seed)
    sc = _silhouette(x_cand, cand, "candidate", warnings, seed)
    res.base_silhouette_score_10d = _f(sb)
    res.candidate_silhouette_score_10d = _f(sc)
    if sb is not None and sc is not None:
        res.silhouette_score_delta = _f(sc - sb)
    return res


def compare_embeddings(
    base_embeddings, candidate_embeddings,
    labels_base=None, labels_candidate=None,
    embeddings_10d_base=None, embeddings_10d_candidate=None,
    k: int = DEFAULT_NEIGHBOR_K,
    delta_thresholds: Sequence[float] = DEFAULT_DELTA_THRESHOLDS,
    max_neighbor_items: int = MAX_NEIGHBOR_ITEMS,
    seed: int = 0,
) -> EmbeddingComparison:
    """Compare base vs candidate embeddings; metric edge cases become warnings, not errors."""
    thresholds = tuple(float(t) for t in delta_thresholds)
    warnings: list[str] = []
    base = _as_matrix(base_embeddings, "base_embeddings", warnings)
    cand = _as_matrix(candidate_embeddings, "candidate_embeddings", warnings)
    if base is None or cand is None:
        return _empty(warnings, thresholds)
    if base.shape != cand.shape:
        warnings.append("base_embeddings and candidate_embeddings must have the same shape.")
        return _empty(warnings, thresholds)
    n = base.shape[0]
    if n == 0:
        return _empty(warnings, thresholds)

    bn = _row_normalize(base, "base_embeddings", warnings)
    cn = _row_normalize(cand, "candidate_embeddings", warnings)
    sims = np.clip(np.sum(bn * cn, axis=1), -1.0, 1.0)
    deltas = 1.0 - sims
    return EmbeddingComparison(
        n_requirements=n,
        per_requirement=[RequirementDelta(i, _f(s), _f(d))
                         for i, (s, d) in enumerate(zip(sims, deltas, strict=True))],
        aggregate=Aggregate(_f(np.mean(sims)), _f(np.median(sims)), _f(np.min(sims)),
                            _f(np.max(sims)), _f(np.mean(deltas))),
        delta_threshold_counts={f"{t:.2f}": int(np.sum(deltas > t)) for t in thresholds},
        nearest_neighbor_preservation=_neighbor_preservation(
            bn, cn, k, max_neighbor_items, warnings),
        cluster_impact=_cluster_impact(labels_base, labels_candidate, embeddings_10d_base,
                                       embeddings_10d_candidate, n, warnings, seed),
        warnings=warnings,
    )


# --- ablation ---

@dataclass
class VariantReport:
    embedding_shape: list[int]
    n_clusters: int = 0
    noise_count: int = 0
    noise_rate: float = 0.0
    silhouette_score_10d: float | None = None
    duration_ms: dict[str, float] = field(default_factory=dict)
    graph_nodes: int = 0
    graph_edges: int = 0
    cluster_info: dict[int, dict] = field(default_factory=dict)


@dataclass
class AblationReport:
    mode: str
    n_requirements: int
    base: VariantReport
    enriched: VariantReport
    comparison: EmbeddingComparison
    warnings: list[str]


def build_embedding_texts(originals: Sequence[str], enriched: Sequence[str | None] | None,
                          mode: str) -> list[str]:
    """Deterministic model input per mode; missing enrichment falls back to the original."""
    return build_texts(originals, mode, enriched)[0]


def _empty_variant(n: int, shape: list[int], durations: dict[str, float]) -> VariantReport:
    return VariantReport(embedding_shape=shape, noise_count=n, noise_rate=1.0 if n else 0.0,
                         duration_ms=durations)


def _run_variant(name: str, texts: list[str], X: np.ndarray, params: ClusteringParams,
                 seed: int, threads: int, durations: dict[str, float],
                 warnings: list[str]):
    n = len(texts)
    shape = [int(v) for v in X.shape]
    none = np.full(n, -1, dtype=np.int64)
    if n == 0:
        warnings.append(f"{name} ablation skipped because there are no requirements.")
        return _empty_variant(n, shape, durations), none, None
    if n < 4:
        warnings.append(f"{name} UMAP/HDBSCAN skipped because at least 4 requirements are needed.")
        return _empty_variant(n, shape, durations), none, None
    if params.min_cluster_size is not None and params.min_cluster_size > n:
        warnings.append(f"{name} clustering skipped because min_cluster_size exceeds requirement count.")
        return _empty_variant(n, shape, durations), none, None

    t = time.perf_counter()
    res = pipeline.run(X, texts, params, seed=seed, threads=threads)
    durations["pipeline"] = elapsed_ms(t)
    labels = res.labels
    k = len(L.cluster_ids(labels))
    noise = int((labels == -1).sum())
    if k == 0:
        warnings.append(f"{name} labeling skipped because no non-noise clusters were found.")
    sil = _silhouette(res.x10, labels, name, warnings, seed)
    return VariantReport(
        embedding_shape=shape, n_clusters=k, noise_count=noise, noise_rate=_f(noise / n),
        silhouette_score_10d=_f(sil), duration_ms=durations,
        graph_nodes=n, graph_edges=len(res.edges),
        cluster_info={c.cluster_id: {"label": c.label, "keywords": c.keywords, "size": c.size}
                      for c in res.clusters},
    ), labels, res.x10


def run_ablation(
    texts: Sequence[str], enriched: Sequence[str | None] | None, params: ClusteringParams,
    embed_fn: Callable[[list[str]], np.ndarray], seed: int, threads: int = 1,
) -> AblationReport:
    """Base-vs-target embedding ablation. params.embedding_mode picks the target mode."""
    mode = params.embedding_mode
    if mode not in ("base", "enriched", "hybrid"):
        raise Invalid("Invalid embedding mode. Use base, enriched, or hybrid.")
    texts = [str(t) for t in texts]
    n = len(texts)
    if enriched is not None and len(enriched) != n:
        raise Invalid("texts and enriched must have the same length.")

    warnings: list[str] = []
    if mode != "base":
        missing = n if enriched is None else sum(1 for e in enriched if e is None or not str(e).strip())
        if missing:
            warnings.append(f"{missing} enriched texts were missing; fallback to base text was used.")

    bd: dict[str, float] = {}
    td: dict[str, float] = {}
    t = time.perf_counter()
    base_X = np.asarray(embed_fn(build_embedding_texts(texts, None, "base")), dtype=np.float32)
    bd["embedding"] = elapsed_ms(t)
    t = time.perf_counter()
    target_X = np.asarray(embed_fn(build_embedding_texts(texts, enriched, mode)), dtype=np.float32)
    td["embedding"] = elapsed_ms(t)

    base_r, base_l, base_10 = _run_variant("base", texts, base_X, params, seed, threads, bd, warnings)
    tgt_r, tgt_l, tgt_10 = _run_variant(mode, texts, target_X, params, seed, threads, td, warnings)
    cmp = compare_embeddings(base_X, target_X, base_l, tgt_l, base_10, tgt_10, seed=seed)
    warnings.extend(cmp.warnings)
    return AblationReport(mode, n, base_r, tgt_r, cmp, warnings)
