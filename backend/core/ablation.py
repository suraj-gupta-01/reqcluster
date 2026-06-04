from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
from sklearn.metrics import silhouette_score

from .clustering import cluster_requirements
from .domain_embeddings import DomainEmbeddingConfig, EmbeddingMode, generate_domain_embeddings
from .embedding_comparison import compare_embeddings
from .graph import build_similarity_graph
from .labeling import label_clusters
from .profiling import record_duration_ms
from .reduction import reduce_embeddings


def _safe_float(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), 6)


def _validate_threshold(value: float) -> None:
    if not isinstance(value, (int, float)) or not (0.0 <= float(value) <= 1.0):
        raise ValueError("similarity_threshold must be between 0.0 and 1.0.")


def _missing_enriched_count(enriched_texts: Sequence[str | None] | None, n: int) -> int:
    if enriched_texts is None:
        return n
    count = 0
    for text in enriched_texts:
        if text is None or str(text).strip() == "":
            count += 1
    return count


def _silhouette_or_none(
    embeddings_10d: np.ndarray | None,
    labels: np.ndarray | None,
    variant_name: str,
    warnings: list[str],
) -> float | None:
    if embeddings_10d is None or labels is None:
        return None
    mask = labels != -1
    if int(mask.sum()) < 3:
        warnings.append(f"{variant_name} silhouette skipped because too few non-noise points exist.")
        return None
    filtered_labels = labels[mask]
    unique_labels = set(int(v) for v in filtered_labels)
    if len(unique_labels) < 2:
        warnings.append(f"{variant_name} silhouette skipped because fewer than two clusters exist.")
        return None
    if len(unique_labels) >= filtered_labels.shape[0]:
        warnings.append(f"{variant_name} silhouette skipped because every point is its own cluster.")
        return None
    try:
        return float(silhouette_score(embeddings_10d[mask], filtered_labels))
    except Exception:
        warnings.append(f"{variant_name} silhouette calculation failed.")
        return None


def _noise_summary(labels: np.ndarray, n: int) -> tuple[int, int, float]:
    n_clusters = len(set(int(v) for v in labels)) - (1 if -1 in labels else 0)
    noise_count = int(np.sum(labels == -1))
    noise_rate = float(noise_count / n) if n else 0.0
    return n_clusters, noise_count, noise_rate


def _empty_variant_report(n: int, embedding_shape: list[int], durations: Dict[str, float]) -> Dict[str, Any]:
    return {
        "embedding_shape": embedding_shape,
        "n_clusters": 0,
        "noise_count": int(n),
        "noise_rate": 1.0 if n else 0.0,
        "silhouette_score_10d": None,
        "duration_ms": durations,
        "graph": {"nodes": 0, "edges": 0},
        "cluster_info": {},
    }


def _run_variant_pipeline(
    variant_name: str,
    texts: list[str],
    req_ids: list[str],
    embeddings: np.ndarray,
    min_cluster_size: int | None,
    min_samples: int | None,
    similarity_threshold: float,
    random_state: int,
    durations: Dict[str, float],
    warnings: list[str],
) -> tuple[Dict[str, Any], np.ndarray, np.ndarray | None]:
    n = len(texts)
    embedding_shape = [int(v) for v in embeddings.shape]
    if n == 0:
        warnings.append(f"{variant_name} ablation skipped because there are no requirements.")
        return _empty_variant_report(n, embedding_shape, durations), np.empty((0,), dtype=int), None
    if n < 4:
        warnings.append(
            f"{variant_name} UMAP/HDBSCAN skipped because at least 4 requirements are needed."
        )
        labels = np.full(n, -1, dtype=int)
        return _empty_variant_report(n, embedding_shape, durations), labels, None
    if min_cluster_size is not None and min_cluster_size > n:
        warnings.append(
            f"{variant_name} clustering skipped because min_cluster_size exceeds requirement count."
        )
        labels = np.full(n, -1, dtype=int)
        return _empty_variant_report(n, embedding_shape, durations), labels, None

    embeddings_10d: np.ndarray | None = None
    embeddings_2d: np.ndarray | None = None
    try:
        with record_duration_ms(durations, "reduction"):
            embeddings_10d, embeddings_2d = reduce_embeddings(
                embeddings,
                random_state=random_state,
            )
    except Exception:
        warnings.append(f"{variant_name} UMAP reduction failed; downstream stages were skipped.")
        labels = np.full(n, -1, dtype=int)
        return _empty_variant_report(n, embedding_shape, durations), labels, None

    labels = np.full(n, -1, dtype=int)
    probabilities = np.zeros(n, dtype=np.float32)
    try:
        with record_duration_ms(durations, "clustering"):
            labels, probabilities = cluster_requirements(
                embeddings_10d,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples or 3,
            )
            labels = labels.astype(int, copy=False)
            probabilities = probabilities.astype(np.float32, copy=False)
    except Exception:
        warnings.append(f"{variant_name} HDBSCAN clustering failed; treating all points as noise.")

    n_clusters, noise_count, noise_rate = _noise_summary(labels, n)

    cluster_info: Dict[int, Dict[str, Any]] = {}
    if n_clusters > 0:
        try:
            with record_duration_ms(durations, "labeling"):
                cluster_info = label_clusters(texts, labels)
        except Exception:
            warnings.append(f"{variant_name} c-TF-IDF labeling failed.")
    else:
        warnings.append(f"{variant_name} labeling skipped because no non-noise clusters were found.")

    graph_counts = {"nodes": 0, "edges": 0}
    if embeddings_2d is not None:
        try:
            with record_duration_ms(durations, "graph"):
                graph = build_similarity_graph(
                    embeddings=embeddings,
                    texts=texts,
                    labels=labels,
                    umap_2d=embeddings_2d,
                    req_ids=req_ids,
                    threshold=similarity_threshold,
                )
            graph_counts = {
                "nodes": len(graph.get("nodes", [])),
                "edges": len(graph.get("edges", [])),
            }
        except Exception:
            warnings.append(f"{variant_name} similarity graph construction failed.")

    silhouette = _silhouette_or_none(embeddings_10d, labels, variant_name, warnings)
    return (
        {
            "embedding_shape": embedding_shape,
            "n_clusters": int(n_clusters),
            "noise_count": int(noise_count),
            "noise_rate": _safe_float(noise_rate),
            "silhouette_score_10d": _safe_float(silhouette),
            "duration_ms": durations,
            "graph": graph_counts,
            "cluster_info": cluster_info,
        },
        labels,
        embeddings_10d,
    )


def run_embedding_ablation(
    original_texts: Sequence[str],
    enriched_texts: Sequence[str | None] | None,
    requirement_ids: Sequence[str] | None = None,
    mode: EmbeddingMode = EmbeddingMode.HYBRID,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    similarity_threshold: float = 0.65,
    batch_size: int = 64,
    random_state: int = 42,
) -> dict:
    """
    Run a read-only base-vs-domain embedding ablation report.
    """
    target_mode = EmbeddingMode(mode)
    _validate_threshold(similarity_threshold)

    texts = [str(t) for t in original_texts]
    n = len(texts)
    if requirement_ids is None or len(requirement_ids) != n:
        req_ids = [f"REQ-{i + 1:03d}" for i in range(n)]
    else:
        req_ids = [str(rid) for rid in requirement_ids]

    warnings: list[str] = []
    if target_mode != EmbeddingMode.BASE:
        missing = _missing_enriched_count(enriched_texts, n)
        if missing:
            warnings.append(
                f"{missing} enriched texts were missing; fallback_to_base=True was used."
            )

    base_durations: Dict[str, float] = {}
    target_durations: Dict[str, float] = {}

    base_config = DomainEmbeddingConfig(
        mode=EmbeddingMode.BASE,
        batch_size=batch_size,
        fallback_to_base=True,
    )
    target_config = DomainEmbeddingConfig(
        mode=target_mode,
        batch_size=batch_size,
        fallback_to_base=True,
    )

    with record_duration_ms(base_durations, "embedding"):
        base_embeddings = generate_domain_embeddings(
            texts,
            None,
            config=base_config,
        )
    with record_duration_ms(target_durations, "embedding"):
        target_embeddings = generate_domain_embeddings(
            texts,
            enriched_texts,
            config=target_config,
        )

    base_report, base_labels, base_10d = _run_variant_pipeline(
        "base",
        texts,
        req_ids,
        base_embeddings,
        min_cluster_size,
        min_samples,
        similarity_threshold,
        random_state,
        base_durations,
        warnings,
    )
    target_report, target_labels, target_10d = _run_variant_pipeline(
        target_mode.value,
        texts,
        req_ids,
        target_embeddings,
        min_cluster_size,
        min_samples,
        similarity_threshold,
        random_state,
        target_durations,
        warnings,
    )

    comparison = compare_embeddings(
        base_embeddings,
        target_embeddings,
        labels_base=base_labels,
        labels_candidate=target_labels,
        embeddings_10d_base=base_10d,
        embeddings_10d_candidate=target_10d,
    )
    warnings.extend(comparison.get("warnings", []))

    return {
        "mode": target_mode.value,
        "n_requirements": int(n),
        "base": base_report,
        "enriched": target_report,
        "comparison": comparison,
        "warnings": warnings,
    }
