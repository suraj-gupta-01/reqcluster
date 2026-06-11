"""Phase 5 active-learning service.

Closes the human-in-the-loop loop: injects the must-link / cannot-link
constraints captured in Phase 4 into the current clustering (constrained
re-clustering), tracks clustering-quality deltas across iterations, and exposes
the uncertainty-sampling review queue.

Operates on the labels already stored for the session (constraint injection),
so it does not require a UMAP/HDBSCAN re-run.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
from sqlalchemy.orm import Session as DBSession

from core.active_learning import clustering_quality, uncertainty_queue
from core.constrained_clustering import apply_constraints
from core.embeddings import generate_embeddings
from core.labeling import label_clusters
from models.database import (
    Cluster,
    ClusteringIteration,
    ConstraintPair,
    FeedbackCorrection,
    Requirement,
    Session,
)

logger = logging.getLogger(__name__)


class ActiveLearningServiceError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


def _load(db: DBSession, session_id: int) -> Tuple[Session, List[Requirement]]:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise ActiveLearningServiceError(404, f"Session {session_id} not found.")
    if session.status != "done":
        raise ActiveLearningServiceError(400, "Run clustering before active learning.")
    reqs = (
        db.query(Requirement)
        .filter(Requirement.session_id == session_id)
        .order_by(Requirement.id.asc())
        .all()
    )
    if not reqs:
        raise ActiveLearningServiceError(400, "No requirements found for session.")
    return session, reqs


def _load_constraint_index_pairs(
    db: DBSession, session_id: int, db_id_to_index: Dict[int, int]
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    pairs = (
        db.query(ConstraintPair)
        .join(FeedbackCorrection, ConstraintPair.feedback_id == FeedbackCorrection.id)
        .filter(
            ConstraintPair.session_id == session_id,
            FeedbackCorrection.status != "rejected",
        )
        .all()
    )
    must, cannot = [], []
    for p in pairs:
        a = db_id_to_index.get(p.requirement_a_id)
        b = db_id_to_index.get(p.requirement_b_id)
        if a is None or b is None:
            continue
        if p.constraint_type == "must-link":
            must.append((a, b))
        elif p.constraint_type == "cannot-link":
            cannot.append((a, b))
    return must, cannot


def _record_iteration(
    db: DBSession, session_id: int, quality: Dict[str, Any], info: Dict[str, Any]
) -> int:
    last = (
        db.query(ClusteringIteration)
        .filter(ClusteringIteration.session_id == session_id)
        .order_by(ClusteringIteration.iteration.desc())
        .first()
    )
    iteration = (last.iteration + 1) if last else 1
    db.add(
        ClusteringIteration(
            session_id=session_id,
            iteration=iteration,
            n_clusters=quality.get("n_clusters"),
            noise_count=quality.get("noise_count"),
            noise_rate=quality.get("noise_rate"),
            silhouette=quality.get("silhouette"),
            must_link_pairs=info.get("must_link_pairs", 0),
            cannot_link_pairs=info.get("cannot_link_pairs", 0),
            points_moved=info.get("points_moved_must_link", 0)
            + info.get("points_moved_cannot_link", 0),
        )
    )
    return iteration


def run_constrained_reclustering(db: DBSession, session_id: int) -> Dict[str, Any]:
    session, reqs = _load(db, session_id)
    texts = [r.text for r in reqs]
    db_id_to_index = {r.id: i for i, r in enumerate(reqs)}
    base_labels = np.array(
        [r.cluster_id if r.cluster_id is not None else -1 for r in reqs], dtype=np.int32
    )
    probs = np.array([r.membership_prob if r.membership_prob is not None else 1.0 for r in reqs])

    must, cannot = _load_constraint_index_pairs(db, session_id, db_id_to_index)
    if not must and not cannot:
        raise ActiveLearningServiceError(
            400, "No active constraints. Submit feedback corrections first."
        )

    try:
        embeddings = generate_embeddings(texts, batch_size=64, use_cache=True)
    except Exception as exc:
        raise ActiveLearningServiceError(500, f"Failed to generate embeddings: {exc}") from exc

    new_labels, info = apply_constraints(embeddings, base_labels, probs, must, cannot)

    # Persist updated assignments.
    for i, req in enumerate(reqs):
        req.cluster_id = int(new_labels[i])
        req.is_noise = bool(new_labels[i] == -1)

    # Rebuild cluster rows from the new labelling.
    db.query(Cluster).filter(Cluster.session_id == session_id).delete()
    cluster_info = label_clusters(texts, new_labels)
    for cid, meta in cluster_info.items():
        db.add(
            Cluster(
                session_id=session_id,
                cluster_id=cid,
                label=meta["label"],
                keywords=meta["keywords"],
                size=meta["size"],
            )
        )

    quality = clustering_quality(embeddings, new_labels)
    session.total_clusters = quality["n_clusters"]
    session.noise_count = quality["noise_count"]
    iteration = _record_iteration(db, session_id, quality, info)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise ActiveLearningServiceError(500, "Failed to persist constrained clustering.")

    return {
        "session_id": session_id,
        "iteration": iteration,
        "constraints": info,
        "quality": quality,
    }


def get_uncertainty_queue(db: DBSession, session_id: int, top_k: int = 20) -> Dict[str, Any]:
    _session, reqs = _load(db, session_id)
    texts = [r.text for r in reqs]
    req_ids = [r.req_id or f"REQ-{i + 1:03d}" for i, r in enumerate(reqs)]
    labels = np.array(
        [r.cluster_id if r.cluster_id is not None else -1 for r in reqs], dtype=np.int32
    )
    probs = np.array([r.membership_prob if r.membership_prob is not None else 1.0 for r in reqs])
    queue = uncertainty_queue(labels, probs, texts, req_ids, top_k=top_k)
    # Attach the requirement DB id so the UI can submit feedback directly.
    by_index = {i: r.id for i, r in enumerate(reqs)}
    for item in queue:
        item["requirement_id"] = by_index.get(item["index"])
    return {"session_id": session_id, "queue": queue, "count": len(queue)}


def get_quality_history(db: DBSession, session_id: int) -> Dict[str, Any]:
    rows = (
        db.query(ClusteringIteration)
        .filter(ClusteringIteration.session_id == session_id)
        .order_by(ClusteringIteration.iteration.asc())
        .all()
    )
    history = [
        {
            "iteration": r.iteration,
            "n_clusters": r.n_clusters,
            "noise_count": r.noise_count,
            "noise_rate": r.noise_rate,
            "silhouette": r.silhouette,
            "must_link_pairs": r.must_link_pairs,
            "cannot_link_pairs": r.cannot_link_pairs,
            "points_moved": r.points_moved,
            "created_at": r.created_at,
        }
        for r in rows
    ]
    return {"session_id": session_id, "history": history, "count": len(history)}
