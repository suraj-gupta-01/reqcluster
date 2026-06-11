"""ClusterLLM coherence scoring and rationale generation.

This module provides LLM-powered (or mock) cluster quality assessment:
- Coherence scoring per cluster
- Merge/split rationale generation
- Cluster summary paragraph generation

The mock provider uses deterministic algorithms (average pairwise cosine
similarity) instead of LLM calls, matching the Phase 2 mock-first approach.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Protocol

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class CoherenceResult:
    """Coherence assessment for a single cluster."""

    cluster_id: int
    coherence_score: float
    top_keywords: List[str]
    size: int
    assessment: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ClusterSummary:
    """Generated summary paragraph for a cluster."""

    cluster_id: int
    summary: str
    representative_count: int

    def to_dict(self) -> dict:
        return asdict(self)


class ClusterRefinementProvider(Protocol):
    """Protocol for cluster refinement providers."""

    name: str

    def score_coherence(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        labels: np.ndarray,
        cluster_id: int,
        keywords: List[str],
    ) -> CoherenceResult: ...

    def generate_merge_rationale(
        self,
        cluster_a_label: str,
        cluster_b_label: str,
        centroid_similarity: float,
        silhouette_delta: float,
        coherence_a: float,
        coherence_b: float,
    ) -> str: ...

    def generate_split_rationale(
        self,
        cluster_label: str,
        bimodality_score: float,
        bic_improvement: float,
        spread_score: float,
        sub_cluster_sizes: List[int],
    ) -> str: ...

    def generate_cluster_summary(
        self,
        cluster_label: str,
        keywords: List[str],
        representative_texts: List[str],
    ) -> str: ...


class MockClusterRefinementProvider:
    """Deterministic, offline provider for cluster refinement.

    Uses algorithmic coherence scoring (average pairwise cosine similarity)
    and template-based rationale/summary generation. No LLM calls.
    """

    name = "mock"

    def score_coherence(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        labels: np.ndarray,
        cluster_id: int,
        keywords: List[str],
    ) -> CoherenceResult:
        mask = labels == cluster_id
        cluster_emb = embeddings[mask]
        n = int(cluster_emb.shape[0])

        if n < 2:
            coherence = 1.0
        else:
            # Exact pairwise coherence is O(n^2) in time and memory; cap to a
            # representative random sample so very large clusters (2000+ members
            # at 35k) score in milliseconds. The mean pairwise similarity is
            # well estimated from a few hundred members.
            emb = cluster_emb
            if n > 400:
                idx = np.random.default_rng(42).choice(n, 400, replace=False)
                emb = cluster_emb[idx]
            m = int(emb.shape[0])
            sim = cosine_similarity(emb)
            upper_sum = float(np.triu(sim, k=1).sum())
            n_pairs = m * (m - 1) / 2
            coherence = upper_sum / n_pairs if n_pairs > 0 else 1.0

        coherence = round(min(max(coherence, 0.0), 1.0), 6)

        if coherence >= 0.8:
            assessment = "Highly coherent cluster with strong semantic consistency."
        elif coherence >= 0.6:
            assessment = "Moderately coherent cluster. Some requirements may be loosely related."
        elif coherence >= 0.4:
            assessment = "Low coherence. Consider splitting into more focused sub-clusters."
        else:
            assessment = "Very low coherence. This cluster likely contains unrelated requirements."

        return CoherenceResult(
            cluster_id=cluster_id,
            coherence_score=coherence,
            top_keywords=list(keywords[:5]),
            size=n,
            assessment=assessment,
        )

    def generate_merge_rationale(
        self,
        cluster_a_label: str,
        cluster_b_label: str,
        centroid_similarity: float,
        silhouette_delta: float,
        coherence_a: float,
        coherence_b: float,
    ) -> str:
        parts = [
            f'Clusters "{cluster_a_label}" and "{cluster_b_label}" share high centroid '
            f"similarity ({centroid_similarity:.3f}).",
        ]

        if silhouette_delta > 0:
            parts.append(
                f"Merging improves the overall silhouette score by {silhouette_delta:.4f}, "
                "indicating better-defined cluster boundaries."
            )
        elif silhouette_delta == 0:
            parts.append("Merging has a neutral effect on the silhouette score.")
        else:
            parts.append(
                f"Merging decreases the silhouette score by {abs(silhouette_delta):.4f}. "
                "Review the clusters carefully before merging."
            )

        avg_coherence = (coherence_a + coherence_b) / 2
        if avg_coherence >= 0.7:
            parts.append(
                "Both clusters have high internal coherence, suggesting they cover "
                "closely related functional domains."
            )
        else:
            parts.append(
                "One or both clusters have moderate coherence. Merging may improve "
                "or reduce overall coherence depending on the requirements."
            )

        return " ".join(parts)

    def generate_split_rationale(
        self,
        cluster_label: str,
        bimodality_score: float,
        bic_improvement: float,
        spread_score: float,
        sub_cluster_sizes: List[int],
    ) -> str:
        parts = [
            f'Cluster "{cluster_label}" shows a bimodal distribution '
            f"(bimodality score: {bimodality_score:.3f}, BIC improvement: {bic_improvement:.1f}).",
        ]

        if len(sub_cluster_sizes) == 2:
            parts.append(
                f"The two identified sub-groups contain {sub_cluster_sizes[0]} and "
                f"{sub_cluster_sizes[1]} requirements respectively."
            )

        if spread_score >= 0.3:
            parts.append(
                f"The cluster has high internal spread ({spread_score:.3f}), "
                "supporting the case for splitting."
            )

        parts.append(
            "Splitting would create more focused clusters with tighter semantic coherence."
        )

        return " ".join(parts)

    def generate_cluster_summary(
        self,
        cluster_label: str,
        keywords: List[str],
        representative_texts: List[str],
    ) -> str:
        parts = [f'Cluster "{cluster_label}"']

        if keywords:
            parts.append(
                f"covers requirements related to {', '.join(keywords[:5])}"
            )

        parts[0] = parts[0] + " " + (parts.pop(1) if len(parts) > 1 else "contains related requirements") + "."

        if representative_texts:
            parts.append(
                "Representative requirements include: "
                + "; ".join(f'"{text[:120]}"' for text in representative_texts[:3])
                + "."
            )

        return " ".join(parts)


class LLMClusterRefinementProvider:
    """LLM-backed refinement provider for narrative output.

    Coherence scoring stays deterministic (it is a numeric measure, not prose).
    Merge/split rationales and cluster summaries are generated by an on-prem or
    OpenAI-compatible LLM (e.g. Qwen via Ollama). Any provider error falls back
    to the deterministic template so the pipeline never fails on LLM issues.
    """

    name = "llm"

    def __init__(self, provider_name: str = "local") -> None:
        self._provider_name = provider_name
        self._fallback = MockClusterRefinementProvider()

    def score_coherence(
        self,
        embeddings: np.ndarray,
        texts: List[str],
        labels: np.ndarray,
        cluster_id: int,
        keywords: List[str],
    ) -> CoherenceResult:
        return self._fallback.score_coherence(embeddings, texts, labels, cluster_id, keywords)

    def _complete(self, prompt: str, max_chars: int = 700) -> Optional[str]:
        from . import providers

        try:
            return providers.generate_completion(
                prompt, self._provider_name, max_chars=max_chars
            )
        except Exception:  # configuration or transport failure -> deterministic fallback
            return None

    def generate_merge_rationale(
        self,
        cluster_a_label: str,
        cluster_b_label: str,
        centroid_similarity: float,
        silhouette_delta: float,
        coherence_a: float,
        coherence_b: float,
    ) -> str:
        prompt = (
            "Two requirement clusters may be merged. Explain in 2-3 sentences whether merging is "
            "advisable, citing the metrics.\n"
            f"Cluster A: {cluster_a_label}\nCluster B: {cluster_b_label}\n"
            f"Centroid cosine similarity: {centroid_similarity:.3f}\n"
            f"Silhouette delta if merged: {silhouette_delta:.4f}\n"
            f"Coherence A: {coherence_a:.3f}, Coherence B: {coherence_b:.3f}"
        )
        return self._complete(prompt) or self._fallback.generate_merge_rationale(
            cluster_a_label, cluster_b_label, centroid_similarity,
            silhouette_delta, coherence_a, coherence_b,
        )

    def generate_split_rationale(
        self,
        cluster_label: str,
        bimodality_score: float,
        bic_improvement: float,
        spread_score: float,
        sub_cluster_sizes: List[int],
    ) -> str:
        prompt = (
            "A requirement cluster may be split into two. Explain in 2-3 sentences whether "
            "splitting is advisable, citing the metrics.\n"
            f"Cluster: {cluster_label}\n"
            f"Bimodality score: {bimodality_score:.3f}\nBIC improvement: {bic_improvement:.1f}\n"
            f"Spread score: {spread_score:.3f}\nSub-cluster sizes: {sub_cluster_sizes}"
        )
        return self._complete(prompt) or self._fallback.generate_split_rationale(
            cluster_label, bimodality_score, bic_improvement, spread_score, sub_cluster_sizes,
        )

    def generate_cluster_summary(
        self,
        cluster_label: str,
        keywords: List[str],
        representative_texts: List[str],
    ) -> str:
        reps = "\n".join(f"- {t[:200]}" for t in representative_texts[:3])
        prompt = (
            "Write a 2-3 sentence summary of this engineering requirement cluster for a review "
            "document.\n"
            f"Cluster label: {cluster_label}\n"
            f"Keywords: {', '.join(keywords[:6])}\n"
            f"Representative requirements:\n{reps}"
        )
        return self._complete(prompt) or self._fallback.generate_cluster_summary(
            cluster_label, keywords, representative_texts,
        )


def get_refinement_provider(provider_name: str = "mock") -> ClusterRefinementProvider:
    """Get a cluster refinement provider by name.

    ``mock`` (default) uses deterministic templates; ``local`` / ``openai`` route
    narrative output through an on-prem or OpenAI-compatible LLM, falling back to
    the deterministic templates if the LLM is unavailable.
    """
    name = (provider_name or "mock").strip().lower()
    if name in {"mock", "offline", "test"}:
        return MockClusterRefinementProvider()
    if name in {"local", "local-llm", "local_llm", "openai", "openai-compatible", "openai_compatible"}:
        return LLMClusterRefinementProvider(name)
    raise ValueError(f"Unsupported refinement provider: {provider_name}")


def score_all_clusters(
    embeddings: np.ndarray,
    texts: List[str],
    labels: np.ndarray,
    cluster_info: Dict[int, Dict],
    provider_name: str = "mock",
) -> Dict[int, CoherenceResult]:
    """Score coherence for all clusters.

    Args:
        embeddings: Original 384-dim embeddings.
        texts: Requirement texts.
        labels: Cluster labels.
        cluster_info: Dict from label_clusters() with {label, keywords, size}.
        provider_name: Refinement provider to use.

    Returns:
        Dict mapping cluster_id → CoherenceResult.
    """
    provider = get_refinement_provider(provider_name)
    results: Dict[int, CoherenceResult] = {}

    unique_clusters = sorted(set(int(l) for l in labels if l != -1))
    for cluster_id in unique_clusters:
        info = cluster_info.get(cluster_id, {})
        keywords = info.get("keywords", [])
        results[cluster_id] = provider.score_coherence(
            embeddings, texts, labels, cluster_id, keywords
        )

    return results
