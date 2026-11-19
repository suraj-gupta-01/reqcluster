"""Merge/split suggestions and cluster representatives. Pure; seed is an argument."""
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture

from . import labels as L
from . import sampling

COHERENCE_CAP = 400
SIL_SAMPLE = 2000


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0, n, 1.0)


# --- merge ---

@dataclass(frozen=True)
class MergeCandidate:
    cluster_a: int
    cluster_b: int
    centroid_similarity: float


@dataclass(frozen=True)
class MergeScore:
    cluster_a: int
    cluster_b: int
    silhouette_before: float
    silhouette_after: float
    silhouette_delta: float


@dataclass(frozen=True)
class MergeSuggestion:
    cluster_a: int
    cluster_b: int
    cluster_a_size: int
    cluster_b_size: int
    centroid_similarity: float
    silhouette_delta: float
    coherence_a: float
    coherence_b: float
    rank: int


def unit_centroids(X: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    ids, C = L.centroids(X, labels)
    return dict(zip(ids, _unit(C), strict=True)) if ids else {}


def pairwise_similarity(centroids: dict[int, np.ndarray]) -> list[MergeCandidate]:
    ids = sorted(centroids)
    if len(ids) < 2:
        return []
    S = cosine_similarity(np.array([centroids[c] for c in ids]))
    out = [MergeCandidate(ids[i], ids[j], float(S[i, j]))
           for i in range(len(ids)) for j in range(i + 1, len(ids))]
    out.sort(key=lambda c: c.centroid_similarity, reverse=True)
    return out


def coherence(X: np.ndarray, labels: np.ndarray, cluster_id: int, seed: int = 0) -> float:
    """Mean pairwise cosine within a cluster (sampled above COHERENCE_CAP points)."""
    E = X[labels == cluster_id]
    n = len(E)
    if n < 2:
        return 1.0
    if n > COHERENCE_CAP:
        E = E[np.random.default_rng(seed).choice(n, COHERENCE_CAP, replace=False)]
        n = COHERENCE_CAP
    return float(np.triu(cosine_similarity(E), k=1).sum() / (n * (n - 1) / 2))


def score_coherence(X: np.ndarray, labels: np.ndarray, cluster_id: int,
                    keywords: list[str]) -> dict:
    n = int((labels == cluster_id).sum())
    score = round(min(max(coherence(X, labels, cluster_id, seed=42), 0.0), 1.0), 6)
    # MiniLM calibration (refine_calibration.py): gold median .39-.48, merged pairs .28-.36, random .20-.24
    if score >= 0.45:
        text = "Highly coherent cluster with strong semantic consistency."
    elif score >= 0.38:
        text = "Moderately coherent cluster. Some requirements may be loosely related."
    elif score >= 0.28:
        text = "Low coherence. Consider splitting into more focused sub-clusters."
    else:
        text = "Very low coherence. This cluster likely contains unrelated requirements."
    return {"cluster_id": cluster_id, "coherence_score": score,
            "top_keywords": list(keywords[:5]), "size": n, "assessment": text}


def score_all_clusters(X: np.ndarray, labels: np.ndarray,
                       keywords: dict[int, list[str]]) -> list[dict]:
    return [score_coherence(X, labels, int(c), keywords.get(int(c), []))
            for c in sorted(set(labels.tolist())) if c != -1]


def evaluate_merge(x10: np.ndarray, labels: np.ndarray, a: int, b: int,
                   seed: int = 0) -> MergeScore | None:
    ids = set(L.cluster_ids(labels))
    if a not in ids or b not in ids or len(ids) < 3:
        return None
    m = L.clustered_mask(labels)
    lab = labels[m].copy()
    if len(lab) < 3:
        return None
    X = x10[m]
    merged = lab.copy()
    merged[merged == b] = a
    # same seeded sample before/after keeps the delta fair
    before = sampling.silhouette(X, lab, SIL_SAMPLE, seed)
    after = sampling.silhouette(X, merged, SIL_SAMPLE, seed)
    if before is None or after is None:
        return None
    return MergeScore(a, b, round(before, 6), round(after, 6), round(after - before, 6))


def suggest_merges(embeddings: np.ndarray, x10: np.ndarray, labels: np.ndarray,
                   top_n: int = 5, sim_threshold: float = 0.75,
                   seed: int = 0) -> list[MergeSuggestion]:
    labels = np.asarray(labels)
    cands = [c for c in pairwise_similarity(unit_centroids(embeddings, labels))
             if c.centroid_similarity >= sim_threshold]
    cands = cands[: max(top_n * 3, 15)]
    out = []
    for c in cands:
        ms = evaluate_merge(x10, labels, c.cluster_a, c.cluster_b, seed)
        if ms is None or ms.silhouette_delta <= 0:
            continue
        out.append(MergeSuggestion(
            c.cluster_a, c.cluster_b,
            int((labels == c.cluster_a).sum()), int((labels == c.cluster_b).sum()),
            round(c.centroid_similarity, 6), round(ms.silhouette_delta, 6),
            round(coherence(embeddings, labels, c.cluster_a, seed), 6),
            round(coherence(embeddings, labels, c.cluster_b, seed), 6), 0))
    out.sort(key=lambda s: (s.silhouette_delta, s.centroid_similarity), reverse=True)
    return [MergeSuggestion(**{**s.__dict__, "rank": i + 1}) for i, s in enumerate(out[:top_n])]


# --- split ---

@dataclass(frozen=True)
class BimodalityResult:
    cluster_id: int
    is_bimodal: bool
    bic_1_component: float
    bic_2_component: float
    bic_improvement: float
    sub_cluster_sizes: list[int]
    sub_labels: list[int]
    bimodality_score: float


@dataclass(frozen=True)
class SplitScore:
    cluster_id: int
    silhouette_before: float
    silhouette_after: float
    silhouette_delta: float


@dataclass(frozen=True)
class SplitSuggestion:
    cluster_id: int
    cluster_size: int
    spread_score: float
    bimodality_score: float
    bic_improvement: float
    silhouette_delta: float
    sub_cluster_sizes: list[int]
    sub_labels: list[int]
    rank: int


def cluster_spread(X: np.ndarray, labels: np.ndarray, cluster_id: int) -> float:
    """Std of cosine distance to the cluster centroid, clamped to [0, 1]."""
    E = X[labels == cluster_id]
    if len(E) < 2:
        return 0.0
    c = _unit(E.mean(axis=0))
    dist = 1.0 - _unit(E) @ c
    return min(max(float(np.std(dist)), 0.0), 1.0)


def bimodality(x10: np.ndarray, labels: np.ndarray, cluster_id: int,
                    min_cluster_size: int = 6, seed: int = 0) -> BimodalityResult | None:
    """1- vs 2-component GMM BIC on the cluster's reduced embeddings."""
    E = x10[labels == cluster_id]
    if len(E) < min_cluster_size:
        return None
    try:
        g1 = GaussianMixture(n_components=1, random_state=seed, covariance_type="full").fit(E)
        g2 = GaussianMixture(n_components=2, random_state=seed, covariance_type="full").fit(E)
        bic1, bic2 = g1.bic(E), g2.bic(E)
    except Exception:
        return None
    imp = float(bic1 - bic2)
    sub = g2.predict(E)
    sizes = [int((sub == c).sum()) for c in range(2)]
    score = 0.0 if imp <= 0 else min(1.0, imp / (imp + 100.0))
    return BimodalityResult(
        cluster_id, imp > 0 and min(sizes) >= 3, round(float(bic1), 3), round(float(bic2), 3),
        round(imp, 3), sizes, [int(v) for v in sub], round(score, 6))


def evaluate_split(x10: np.ndarray, labels: np.ndarray, cluster_id: int,
                   sub_labels: list[int], seed: int = 0) -> SplitScore | None:
    m = L.clustered_mask(labels)
    lab = labels[m].copy()
    if len(lab) < 3:
        return None
    X = x10[m]
    idx = np.where(lab == cluster_id)[0]
    if len(idx) != len(sub_labels):
        return None
    split = lab.copy()
    sl = np.asarray(sub_labels)
    split[idx[sl == 1]] = int(lab.max()) + 1
    before = sampling.silhouette(X, lab, SIL_SAMPLE, seed)
    after = sampling.silhouette(X, split, SIL_SAMPLE, seed)
    if before is None or after is None:
        return None
    return SplitScore(cluster_id, round(before, 6), round(after, 6), round(after - before, 6))


def suggest_splits(embeddings: np.ndarray, x10: np.ndarray, labels: np.ndarray,
                   top_n: int = 5, spread_threshold: float = 0.3,
                   seed: int = 0) -> list[SplitSuggestion]:
    labels = np.asarray(labels)
    out = []
    for cid, idx in L.groups(labels).items():
        spread = cluster_spread(embeddings, labels, cid)  # spread_threshold is informational only
        size = len(idx)
        bm = bimodality(x10, labels, cid, seed=seed)
        if bm is None or not bm.is_bimodal:
            continue
        sc = evaluate_split(x10, labels, cid, bm.sub_labels, seed)
        if sc is None or sc.silhouette_delta <= 0:
            continue
        out.append(SplitSuggestion(
            cid, size, round(spread, 6), bm.bimodality_score, bm.bic_improvement,
            round(sc.silhouette_delta, 6),
            bm.sub_cluster_sizes, bm.sub_labels, 0))
    out.sort(key=lambda s: (s.silhouette_delta, s.bimodality_score), reverse=True)
    return [SplitSuggestion(**{**s.__dict__, "rank": i + 1}) for i, s in enumerate(out[:top_n])]


# --- representatives ---

@dataclass(frozen=True)
class Representative:
    req_id: str
    text: str
    index: int
    similarity_to_centroid: float
    rank: int


def representatives(embeddings: np.ndarray, texts: list[str], labels: np.ndarray,
                    req_ids: list[str], top_n: int = 3) -> dict[int, list[Representative]]:
    """Per cluster, the top_n requirements closest (cosine) to the centroid."""
    out: dict[int, list[Representative]] = {}
    for cid, idx in L.groups(np.asarray(labels)).items():
        E = embeddings[idx]
        sims = cosine_similarity(E, _unit(E.mean(axis=0, keepdims=True))).ravel()
        top = np.argsort(sims)[::-1][:top_n]
        out[cid] = [
            Representative(
                req_ids[int(idx[j])] if idx[j] < len(req_ids) else f"REQ-{int(idx[j])}",
                texts[int(idx[j])] if idx[j] < len(texts) else "",
                int(idx[j]), round(float(sims[j]), 6), r + 1)
            for r, j in enumerate(top)]
    return out


def summary_texts(embeddings, texts, labels, req_ids, top_n: int = 3) -> dict[int, str]:
    return {cid: "; ".join(r.text.strip() for r in reps if r.text.strip())
            for cid, reps in representatives(embeddings, texts, labels, req_ids, top_n).items()}
