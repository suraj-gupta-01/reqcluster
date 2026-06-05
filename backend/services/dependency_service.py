"""Dependency tree + rationale document service (DP5).

Builds the requirement dependency DAG, generates a human-readable rationale
document (why requirements are grouped + why each dependency exists), persists
both, and exposes retrieval. Narrative prose uses the refinement provider
(deterministic mock by default; on-prem LLM when configured).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session as DBSession

from core.dependency_tree import build_dependency_tree
from core.embeddings import generate_embeddings
from core.representatives import extract_representatives
from llm_services.refinement import get_refinement_provider
from models.database import Cluster, DependencyTree, Requirement, Session

logger = logging.getLogger(__name__)


class DependencyServiceError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


def _load_session(db: DBSession, session_id: int) -> tuple[Session, List[Requirement]]:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise DependencyServiceError(404, f"Session {session_id} not found.")
    if session.status != "done":
        raise DependencyServiceError(
            400, "Run clustering before generating the dependency tree."
        )
    reqs = (
        db.query(Requirement)
        .filter(Requirement.session_id == session_id)
        .order_by(Requirement.id.asc())
        .all()
    )
    if not reqs:
        raise DependencyServiceError(400, "No requirements found for session.")
    return session, reqs


def _build_rationale_document(
    db: DBSession,
    session_id: int,
    reqs: List[Requirement],
    embeddings: np.ndarray,
    labels: np.ndarray,
    req_ids: List[str],
    tree: Dict[str, Any],
    provider_name: str,
) -> Dict[str, Any]:
    """Assemble the DP5 rationale document: grouping + dependency justification."""
    texts = [r.text for r in reqs]
    provider = get_refinement_provider(provider_name)

    clusters = (
        db.query(Cluster)
        .filter(Cluster.session_id == session_id)
        .order_by(Cluster.cluster_id.asc())
        .all()
    )
    reps = extract_representatives(embeddings, texts, labels, req_ids, top_n=3)

    grouping: List[Dict[str, Any]] = []
    for c in clusters:
        keywords = c.keywords if isinstance(c.keywords, list) else []
        rep_texts = [r.text for r in reps.get(c.cluster_id, [])]
        summary = provider.generate_cluster_summary(c.label, keywords, rep_texts[:3])
        grouping.append(
            {
                "cluster_id": c.cluster_id,
                "label": c.label,
                "keywords": keywords,
                "size": c.size,
                "representative_req_ids": [r.req_id for r in reps.get(c.cluster_id, [])],
                "rationale": summary,
            }
        )

    # Edge-level justifications come straight from the deterministic builder; the
    # rationale string already explains each dependency in plain language.
    node_label = {n["id"]: n["node_id"] for n in tree["nodes"]}
    dependencies = [
        {
            "source": e["source"],
            "target": e["target"],
            "source_req_id": node_label.get(e["source"]),
            "target_req_id": node_label.get(e["target"]),
            "relation": e["relation"],
            "weight": e["weight"],
            "justification": e["rationale"],
        }
        for e in tree["edges"]
    ]

    return {
        "session_id": session_id,
        "provider": provider_name,
        "grouping": grouping,
        "dependencies": dependencies,
        "stats": tree.get("stats", {}),
    }


def generate_and_persist_dependencies(
    db: DBSession,
    session_id: int,
    provider_name: str = "mock",
    sim_threshold: float = 0.45,
    top_k: int = 8,
) -> Dict[str, Any]:
    session, reqs = _load_session(db, session_id)
    texts = [r.text for r in reqs]
    req_ids = [r.req_id or f"REQ-{i + 1:03d}" for i, r in enumerate(reqs)]
    labels = np.array(
        [r.cluster_id if r.cluster_id is not None else -1 for r in reqs],
        dtype=np.int32,
    )

    try:
        embeddings = generate_embeddings(texts, batch_size=64, use_cache=True)
    except Exception as exc:
        raise DependencyServiceError(500, f"Failed to generate embeddings: {exc}") from exc

    tree = build_dependency_tree(
        embeddings, texts, req_ids, labels, top_k=top_k, sim_threshold=sim_threshold
    )
    rationale = _build_rationale_document(
        db, session_id, reqs, embeddings, labels, req_ids, tree, provider_name
    )

    row = db.query(DependencyTree).filter(DependencyTree.session_id == session_id).first()
    if row is None:
        row = DependencyTree(session_id=session_id)
        db.add(row)
    row.nodes = tree["nodes"]
    row.edges = tree["edges"]
    row.stats = tree["stats"]
    row.rationale = rationale
    row.provider = provider_name

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise DependencyServiceError(500, "Failed to persist dependency tree.")

    return {
        "session_id": session_id,
        "nodes": tree["nodes"],
        "edges": tree["edges"],
        "stats": tree["stats"],
        "rationale": rationale,
    }


def get_dependencies(db: DBSession, session_id: int) -> Dict[str, Any]:
    row = db.query(DependencyTree).filter(DependencyTree.session_id == session_id).first()
    if row is None:
        raise DependencyServiceError(
            404, "Dependency tree not found. Generate it first."
        )
    return {
        "session_id": session_id,
        "nodes": row.nodes,
        "edges": row.edges,
        "stats": row.stats or {},
        "rationale": row.rationale or {},
    }
