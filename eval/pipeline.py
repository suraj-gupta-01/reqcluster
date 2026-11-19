"""Orchestration pipeline driving the FastAPI TestClient through the evaluation lifecycle."""

from __future__ import annotations

import logging
from typing import Any
import csv
from pathlib import Path

from fastapi.testclient import TestClient

from .config import DatasetConfig
from .deps_metrics import evaluate_dependency_metrics
from .enrich_metrics import evaluate_enrichment_metrics
from .metrics import score_clustering
from .refine_metrics import evaluate_refinement_metrics


logger = logging.getLogger(__name__)


def _norm(text: Any) -> str:
    return " ".join(str(text).split()).casefold()


def _ok(resp: Any, what: str) -> Any:
    if resp.status_code >= 400:
        raise RuntimeError(f"{what} failed ({resp.status_code}): {resp.text}")
    return resp.json()


def _job(resp: Any, what: str) -> dict[str, Any]:
    """Jobs run inline (0 workers), so the POST already returns a terminal Job."""
    job = _ok(resp, what)
    if job.get("status") != "succeeded":
        raise RuntimeError(f"{what} failed: {job.get('error') or job.get('status')}")
    return job


def _ground_truth(csv_path: Path, column: str) -> dict[str, str]:
    """First-occurrence label per normalised text, matching the backend's dedupe key."""
    gt: dict[str, str] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            gt.setdefault(_norm(row.get("text", "")), str(row.get(column) or "").strip())
    return gt


def _fetch_requirements(client: TestClient, sid: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    page = 1
    while True:
        resp = client.get(f"/api/sessions/{sid}/requirements", params={"page_size": 1000, "page": page})
        items.extend(_ok(resp, "Requirements retrieval"))
        if len(items) >= int(resp.headers.get("X-Total-Count", 0)):
            return sorted(items, key=lambda r: r["ordinal"])
        page += 1


def run_dataset_pipeline(
    client: TestClient,
    dataset_config: DatasetConfig,
    provider: str = "mock",
) -> dict[str, Any]:
    """Execute the end-to-end user journey for a single dataset and collect metrics."""
    dataset_name = dataset_config.name
    csv_path = dataset_config.path
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    logger.info("Evaluating dataset '%s' from %s", dataset_name, csv_path)

    with open(csv_path, "rb") as f:
        up = _ok(client.post("/api/sessions", files={"file": (csv_path.name, f, "text/csv")}), "Upload")
    sid = up["session"]["id"]

    _job(client.post(f"/api/sessions/{sid}/cluster", json={}), "Clustering")
    _ok(client.get(f"/api/sessions/{sid}/metrics"), "Metrics retrieval")

    reqs = _fetch_requirements(client, sid)
    gt_map = _ground_truth(csv_path, dataset_config.ground_truth)
    modules = [gt_map.get(_norm(r["text"]), "") for r in reqs]
    # Dicts shaped like the legacy API so deps/refine scorers are unchanged.
    all_reqs = [{**r, "module": m} for r, m in zip(reqs, modules, strict=True)]
    clustering_metrics = score_clustering(reqs, modules)

    _job(client.post(f"/api/sessions/{sid}/dependencies", json={"provider": provider}), "Dependency generation")
    tree = _ok(client.get(f"/api/sessions/{sid}/dependencies"), "Dependency retrieval")
    rid = {r["id"]: r["req_id"] for r in reqs}
    dep_data = {
        "edges": [
            {**e, "source": rid.get(e["source"], ""), "target": rid.get(e["target"], "")}
            for e in tree.get("edges", [])
        ]
    }
    deps_metrics = evaluate_dependency_metrics(
        requirements=all_reqs, dep_response=dep_data, edges_path=dataset_config.edges
    )

    _job(client.post(f"/api/sessions/{sid}/enrich", json={"provider": provider}), "Enrichment")
    enrich_rows = _ok(client.get(f"/api/sessions/{sid}/enrichment"), "Enrichment results retrieval")
    enrich_metrics = evaluate_enrichment_metrics(enrich_rows)

    _job(client.post(f"/api/sessions/{sid}/suggestions", json={"provider": provider, "top_n_merges": 5, "top_n_splits": 5}), "Suggestion generation")
    sugg = _ok(client.get(f"/api/sessions/{sid}/suggestions"), "Suggestion retrieval")
    suggestions_data = {
        "merge_suggestions": [
            {"cluster_a_id": s["payload"].get("cluster_a"), "cluster_b_id": s["payload"].get("cluster_b")}
            for s in sugg if s["kind"] == "merge"
        ],
        "split_suggestions": [{"cluster_id": s["payload"].get("cluster_id")} for s in sugg if s["kind"] == "split"],
    }
    refine_metrics = evaluate_refinement_metrics(requirements=all_reqs, suggestions_data=suggestions_data)

    combined_metrics: dict[str, Any] = {}
    for part in (clustering_metrics, deps_metrics, enrich_metrics, refine_metrics):
        combined_metrics.update(part)
    return {"dataset": dataset_name, "session_id": sid, "status": "success", "metrics": combined_metrics}
