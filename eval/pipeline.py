"""Orchestration pipeline driving the FastAPI TestClient through the evaluation lifecycle."""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi.testclient import TestClient

from .config import DatasetConfig
from .deps_metrics import evaluate_dependency_metrics
from .enrich_metrics import evaluate_enrichment_metrics
from .metrics import extract_clustering_metrics
from .refine_metrics import evaluate_refinement_metrics


logger = logging.getLogger(__name__)


def run_dataset_pipeline(
    client: TestClient,
    dataset_config: DatasetConfig,
    provider: str = "mock",
    max_poll_seconds: int = 120,
) -> Dict[str, Any]:
    """Execute the end-to-end user journey for a single dataset and collect metrics."""
    dataset_name = dataset_config.name
    csv_path = dataset_config.path

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    logger.info("Evaluating dataset '%s' from %s", dataset_name, csv_path)

    # 1. Upload
    with open(csv_path, "rb") as f:
        upload_resp = client.post(
            "/api/upload",
            files={"file": (csv_path.name, f, "text/csv")},
        )
    if upload_resp.status_code != 200:
        raise RuntimeError(
            f"Upload failed ({upload_resp.status_code}): {upload_resp.text}"
        )

    upload_data = upload_resp.json()
    session_id = upload_data["session_id"]
    logger.info("Dataset '%s' uploaded to session_id=%d", dataset_name, session_id)

    # 2. Cluster
    cluster_resp = client.post(
        "/api/cluster",
        json={
            "session_id": session_id,
            "embedding_mode": "base",
            "similarity_threshold": 0.65,
        },
    )
    if cluster_resp.status_code not in (200, 202):
        raise RuntimeError(
            f"Clustering request failed ({cluster_resp.status_code}): {cluster_resp.text}"
        )

    # 3. Poll progress
    start_time = time.time()
    while True:
        prog_resp = client.get(f"/api/progress/{session_id}")
        if prog_resp.status_code == 200:
            prog_data = prog_resp.json()
            step = prog_data.get("step")
            if step == "done":
                break
            if step in ("error", "failed"):
                raise RuntimeError(
                    f"Clustering pipeline failed: {prog_data.get('message')}"
                )

        if time.time() - start_time > max_poll_seconds:
            raise TimeoutError(
                f"Clustering timed out after {max_poll_seconds} seconds for session {session_id}"
            )
        time.sleep(0.5)

    # 4. Fetch clustering validation metrics
    metrics_resp = client.get(f"/api/metrics?session_id={session_id}")
    if metrics_resp.status_code != 200:
        raise RuntimeError(
            f"Metrics retrieval failed ({metrics_resp.status_code}): {metrics_resp.text}"
        )
    clustering_metrics = extract_clustering_metrics(metrics_resp.json())

    # 5. Dependencies inference and evaluation
    dep_gen_resp = client.post(
        "/api/dependencies/generate",
        json={
            "session_id": session_id,
            "provider_name": provider,
            "sim_threshold": 0.45,
            "top_k": 8,
        },
    )
    if dep_gen_resp.status_code != 200:
        raise RuntimeError(
            f"Dependency generation failed ({dep_gen_resp.status_code}): {dep_gen_resp.text}"
        )
    dep_data = dep_gen_resp.json()

    # Get all requirements for reference ground truth & refinement
    reqs_resp = client.get(f"/api/requirements?session_id={session_id}")
    if reqs_resp.status_code != 200:
        raise RuntimeError(
            f"Requirements retrieval failed ({reqs_resp.status_code}): {reqs_resp.text}"
        )
    all_reqs = reqs_resp.json()

    deps_metrics = evaluate_dependency_metrics(
        requirements=all_reqs,
        dep_response=dep_data,
        edges_path=dataset_config.edges,
    )

    # 6. LLM enrichment & evaluation
    enrich_resp = client.post(
        "/api/enrich",
        json={
            "session_id": session_id,
            "provider_name": provider,
            "embedding_mode": "hybrid",
            "batch_size": 16,
            "use_cache": False,
        },
    )
    if enrich_resp.status_code != 200:
        raise RuntimeError(
            f"Enrichment request failed ({enrich_resp.status_code}): {enrich_resp.text}"
        )

    enrich_results_resp = client.get(f"/api/enrich/results?session_id={session_id}")
    if enrich_results_resp.status_code != 200:
        raise RuntimeError(
            f"Enrichment results retrieval failed ({enrich_results_resp.status_code}): {enrich_results_resp.text}"
        )
    enrich_results = enrich_results_resp.json()
    enrich_metrics = evaluate_enrichment_metrics(enrich_results)

    # 7. Refinement suggestions & evaluation
    sugg_gen_resp = client.post(
        "/api/suggestions/generate",
        json={
            "session_id": session_id,
            "provider_name": provider,
            "top_n_merges": 5,
            "top_n_splits": 5,
        },
    )
    if sugg_gen_resp.status_code != 200:
        raise RuntimeError(
            f"Suggestion generation failed ({sugg_gen_resp.status_code}): {sugg_gen_resp.text}"
        )
    suggestions_data = sugg_gen_resp.json()

    refine_metrics = evaluate_refinement_metrics(
        requirements=all_reqs,
        suggestions_data=suggestions_data,
    )

    # Combine all metrics into unified dictionary
    combined_metrics: Dict[str, Any] = {}
    combined_metrics.update(clustering_metrics)
    combined_metrics.update(deps_metrics)
    combined_metrics.update(enrich_metrics)
    combined_metrics.update(refine_metrics)

    return {
        "dataset": dataset_name,
        "session_id": session_id,
        "status": "success",
        "metrics": combined_metrics,
    }
