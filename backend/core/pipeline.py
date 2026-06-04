from typing import List, Optional, Callable, Dict, Any
import logging

from .embeddings import generate_embeddings
from .domain_embeddings import (
    DomainEmbeddingConfig,
    EmbeddingMode,
    generate_domain_embeddings,
)
from .embedding_comparison import compare_embeddings
from .ablation import run_embedding_ablation
from .reduction import reduce_embeddings
from .clustering import cluster_requirements
from .labeling import label_clusters
from .graph import build_similarity_graph

logger = logging.getLogger(__name__)


def run_pipeline(
    texts: List[str],
    req_ids: List[str],
    min_cluster_size: Optional[int] = None,
    min_samples: int = 3,
    similarity_threshold: float = 0.65,
    progress_callback: Optional[Callable[[str, int, str], None]] = None,
    embedding_mode: str = "base",
    enriched_texts: Optional[List[str | None]] = None,
    enable_embedding_comparison: bool = False,
    run_ablation: bool = False,
) -> Dict[str, Any]:
    """
    Run the full Phase 1 pipeline:
    texts -> embeddings -> UMAP -> HDBSCAN -> labels -> graph
    
    Returns complete results dict.
    """

    def step(name: str, pct: int, msg: str):
        if progress_callback:
            progress_callback(name, pct, msg)
        logger.info(f"[{pct}%] {name}: {msg}")

    try:
        mode = EmbeddingMode(embedding_mode)
    except ValueError as exc:
        raise ValueError("Invalid embedding_mode. Use base, enriched, or hybrid.") from exc

    if not (0.0 <= float(similarity_threshold) <= 1.0):
        raise ValueError("similarity_threshold must be between 0.0 and 1.0.")
    if enriched_texts is not None and len(enriched_texts) != len(texts):
        raise ValueError("enriched_texts must match the number of requirement texts.")

    n = len(texts)
    warnings: List[str] = []

    # Step 1: Embeddings
    if mode == EmbeddingMode.BASE:
        step("embedding", 5, f"Generating embeddings for {n} requirements...")
        embeddings = generate_embeddings(
            texts,
            batch_size=64,
            progress_callback=lambda cur, tot: step(
                "embedding", int(5 + 25 * cur / max(tot, 1)), f"Embedding {cur}/{tot}..."
            ),
            use_cache=True,
        )
        step("embedding", 30, f"Embeddings complete: shape {embeddings.shape}")
    else:
        missing_enriched = (
            n
            if enriched_texts is None
            else sum(1 for text in enriched_texts if text is None or str(text).strip() == "")
        )
        if missing_enriched:
            warnings.append(
                f"{missing_enriched} enriched texts were missing; fallback_to_base=True was used."
            )

        step(
            "embedding_enriched",
            5,
            f"Generating {mode.value} embeddings for {n} requirements...",
        )
        config = DomainEmbeddingConfig(mode=mode, batch_size=64, fallback_to_base=True)
        embeddings = generate_domain_embeddings(
            texts,
            enriched_texts,
            config=config,
            progress_callback=lambda cur, tot: step(
                "embedding_enriched",
                int(5 + 25 * cur / max(tot, 1)),
                f"Embedding {cur}/{tot}...",
            ),
            use_cache=True,
        )
        step("embedding_enriched", 30, f"Embeddings complete: shape {embeddings.shape}")

    embedding_comparison = None
    if enable_embedding_comparison:
        step("embedding_base", 31, "Preparing base embeddings for comparison...")
        if mode == EmbeddingMode.BASE:
            base_embeddings = embeddings
        else:
            base_embeddings = generate_embeddings(
                texts,
                batch_size=64,
                progress_callback=lambda cur, tot: step(
                    "embedding_base",
                    int(31 + 4 * cur / max(tot, 1)),
                    f"Base embedding {cur}/{tot}...",
                ),
                use_cache=True,
            )
        step("embedding_comparison", 35, "Comparing base and selected embeddings...")
        embedding_comparison = compare_embeddings(base_embeddings, embeddings)

    # Step 2: UMAP
    step("umap", 36 if enable_embedding_comparison else 32, "Running UMAP dimensionality reduction...")
    embeddings_10d, embeddings_2d = reduce_embeddings(embeddings)
    step("umap", 55, "UMAP complete")

    # Step 3: HDBSCAN
    step("clustering", 57, "Running HDBSCAN clustering...")
    labels, probabilities = cluster_requirements(
        embeddings_10d,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
    )
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = int((labels == -1).sum())
    step("clustering", 70, f"Found {n_clusters} clusters, {noise_count} noise points")

    # Step 4: Labeling
    step("labeling", 72, "Generating cluster labels with c-TF-IDF...")
    cluster_info = label_clusters(texts, labels)
    step("labeling", 82, f"Labeled {len(cluster_info)} clusters")

    # Step 5: Graph
    step("graph", 84, "Building similarity graph...")
    graph_data = build_similarity_graph(
        embeddings=embeddings,
        texts=texts,
        labels=labels,
        umap_2d=embeddings_2d,
        req_ids=req_ids,
        threshold=similarity_threshold,
    )
    step(
        "graph",
        90 if run_ablation else 98,
        f"Graph built: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges",
    )

    ablation_report = None
    if run_ablation:
        step("ablation", 92, "Running embedding ablation analysis...")
        ablation_report = run_embedding_ablation(
            original_texts=texts,
            enriched_texts=enriched_texts,
            requirement_ids=req_ids,
            mode=mode,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            similarity_threshold=similarity_threshold,
            batch_size=64,
            random_state=42,
        )
        step("ablation", 99, "Embedding ablation analysis complete")

    step("done", 100, "Pipeline complete!")

    results = {
        "embeddings": embeddings,
        "embeddings_10d": embeddings_10d,
        "embeddings_2d": embeddings_2d,
        "labels": labels,
        "probabilities": probabilities,
        "cluster_info": cluster_info,
        "graph_data": graph_data,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
        "embedding_mode": mode.value,
    }

    if warnings:
        results["warnings"] = warnings
    if embedding_comparison is not None:
        results["embedding_comparison"] = embedding_comparison
    if ablation_report is not None:
        results["ablation_report"] = ablation_report

    return results
