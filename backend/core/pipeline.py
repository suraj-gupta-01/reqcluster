import numpy as np
from typing import List, Optional, Callable, Dict, Any
import logging

from .embeddings import generate_embeddings
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

    n = len(texts)
    step("embedding", 5, f"Generating embeddings for {n} requirements...")

    # Step 1: Embeddings
    embeddings = generate_embeddings(
        texts,
        batch_size=64,
        progress_callback=lambda cur, tot: step(
            "embedding", int(5 + 25 * cur / max(tot, 1)), f"Embedding {cur}/{tot}..."
        ),
        use_cache=True,
    )
    step("embedding", 30, f"Embeddings complete: shape {embeddings.shape}")

    # Step 2: UMAP
    step("umap", 32, "Running UMAP dimensionality reduction...")
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
    step("graph", 98, f"Graph built: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

    step("done", 100, "Pipeline complete!")

    return {
        "embeddings": embeddings,
        "embeddings_10d": embeddings_10d,
        "embeddings_2d": embeddings_2d,
        "labels": labels,
        "probabilities": probabilities,
        "cluster_info": cluster_info,
        "graph_data": graph_data,
        "n_clusters": n_clusters,
        "noise_count": noise_count,
    }
