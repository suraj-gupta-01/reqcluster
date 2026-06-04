"""Phase 4 human feedback service layer.

Handles feedback corrections submission, approval/rejection state machine,
DB transactions, cluster size adjustments, and constraint pair operations.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session as DBSession

from core.feedback_bridge import generate_constraints_for_correction
from llm_services.feedback_analyst import FeedbackAnalyst
from models.database import Cluster, ConstraintPair, FeedbackCorrection, Requirement


def submit_feedback(db: DBSession, request: Any) -> FeedbackCorrection:
    """Submit a human-in-the-loop requirement cluster correction.

    Updates the requirement's cluster assignment, increments/decrements the
    affected cluster sizes, registers a FeedbackCorrection record, and
    generates must-link/cannot-link constraint pairs.
    """
    session_id = int(request.session_id)
    requirement_id = int(request.requirement_id)
    new_cluster_id = request.new_cluster_id  # None or -1 for noise

    # 1. Fetch requirement
    req = db.query(Requirement).filter(
        Requirement.id == requirement_id,
        Requirement.session_id == session_id,
    ).first()
    if not req:
        raise ValueError(f"Requirement {requirement_id} not found in session {session_id}.")

    old_cluster_id = req.cluster_id
    if old_cluster_id == new_cluster_id:
        raise ValueError("New cluster assignment is identical to the current one.")

    # 2. Extract cluster context for confidence analysis
    target_label = "Noise"
    target_keywords = []
    if new_cluster_id is not None and new_cluster_id != -1:
        target_cluster = db.query(Cluster).filter(
            Cluster.cluster_id == new_cluster_id,
            Cluster.session_id == session_id,
        ).first()
        if not target_cluster:
            raise ValueError(f"Target cluster {new_cluster_id} not found.")
        target_label = target_cluster.label
        target_keywords = target_cluster.keywords or []

    # 3. Analyze annotation comments
    analyst = FeedbackAnalyst()
    comments = (request.comments or "").strip()
    confidence = request.confidence_score

    if comments and (confidence is None or confidence == 1.0):
        # Dynamically compute confidence from keyword overlap
        confidence = analyst.evaluate_comment_confidence(
            comments, target_label, target_keywords
        )
    elif confidence is None:
        confidence = 1.0

    # 4. Mutate requirement assignment
    req.cluster_id = new_cluster_id

    # 5. Mutate cluster sizes
    if old_cluster_id is not None and old_cluster_id != -1:
        old_c = db.query(Cluster).filter(
            Cluster.cluster_id == old_cluster_id,
            Cluster.session_id == session_id,
        ).first()
        if old_c:
            old_c.size = max(0, old_c.size - 1)

    if new_cluster_id is not None and new_cluster_id != -1:
        new_c = db.query(Cluster).filter(
            Cluster.cluster_id == new_cluster_id,
            Cluster.session_id == session_id,
        ).first()
        if new_c:
            new_c.size += 1

    # 6. Persist FeedbackCorrection
    correction = FeedbackCorrection(
        session_id=session_id,
        requirement_id=requirement_id,
        previous_cluster_id=old_cluster_id,
        new_cluster_id=new_cluster_id,
        confidence_score=confidence,
        comments=comments,
        applied_by=request.applied_by or "Expert Analyst",
        status="pending",
    )
    db.add(correction)
    db.flush()

    # 7. Generate ConstraintPairs
    generate_constraints_for_correction(
        db,
        session_id=session_id,
        feedback_id=correction.id,
        requirement_id=requirement_id,
        old_cluster_id=old_cluster_id,
        new_cluster_id=new_cluster_id,
    )

    try:
        db.commit()
    except Exception as exc:
        db.rollback()
        raise exc

    return correction


def review_feedback(
    db: DBSession,
    session_id: int,
    feedback_id: int,
    status: str,
) -> FeedbackCorrection:
    """Approve or reject a pending feedback correction.

    - Approved corrections simply transition to 'approved' state.
    - Rejected corrections revert the requirement's cluster assignment,
      restore previous cluster sizes, and delete the associated constraint pairs.
    """
    correction = db.query(FeedbackCorrection).filter(
        FeedbackCorrection.id == feedback_id,
        FeedbackCorrection.session_id == session_id,
    ).first()
    if not correction:
        raise ValueError(f"Feedback correction {feedback_id} not found.")

    if correction.status != "pending":
        raise ValueError(f"Feedback correction is already {correction.status}.")

    if status not in {"approved", "rejected"}:
        raise ValueError(f"Invalid review status: {status}")

    if status == "approved":
        correction.status = "approved"
        correction.updated_at = datetime.utcnow()
        db.commit()
    else:
        # Rollback cluster assignment
        req = db.query(Requirement).filter(
            Requirement.id == correction.requirement_id,
            Requirement.session_id == session_id,
        ).first()
        if req:
            current_cluster_id = req.cluster_id
            req.cluster_id = correction.previous_cluster_id

            # Revert cluster sizes
            if current_cluster_id is not None and current_cluster_id != -1:
                c = db.query(Cluster).filter(
                    Cluster.cluster_id == current_cluster_id,
                    Cluster.session_id == session_id,
                ).first()
                if c:
                    c.size = max(0, c.size - 1)

            if correction.previous_cluster_id is not None and correction.previous_cluster_id != -1:
                c = db.query(Cluster).filter(
                    Cluster.cluster_id == correction.previous_cluster_id,
                    Cluster.session_id == session_id,
                ).first()
                if c:
                    c.size += 1

        # Delete generated constraint pairs
        db.query(ConstraintPair).filter(
            ConstraintPair.feedback_id == feedback_id,
            ConstraintPair.session_id == session_id,
        ).delete()

        correction.status = "rejected"
        correction.updated_at = datetime.utcnow()
        db.commit()

    return correction


def get_feedback_queue(
    db: DBSession,
    session_id: int,
    status_filter: Optional[str] = None,
) -> List[FeedbackCorrection]:
    """Retrieve adjustments from the feedback corrections queue."""
    query = db.query(FeedbackCorrection).filter(
        FeedbackCorrection.session_id == session_id
    )
    if status_filter:
        query = query.filter(FeedbackCorrection.status == status_filter)
    return query.order_by(FeedbackCorrection.created_at.desc()).all()


def export_feedback_csv(db: DBSession, session_id: int) -> str:
    """Export all corrections for a session to a CSV string."""
    corrections = get_feedback_queue(db, session_id)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id",
        "requirement_id",
        "previous_cluster_id",
        "new_cluster_id",
        "confidence_score",
        "comments",
        "applied_by",
        "status",
        "created_at",
    ])
    for c in corrections:
        writer.writerow([
            c.id,
            c.requirement_id,
            c.previous_cluster_id,
            c.new_cluster_id,
            c.confidence_score,
            c.comments,
            c.applied_by,
            c.status,
            c.created_at.isoformat(),
        ])
    return output.getvalue()


def export_feedback_json(db: DBSession, session_id: int) -> str:
    """Export all corrections for a session to a JSON string."""
    corrections = get_feedback_queue(db, session_id)
    data = []
    for c in corrections:
        data.append({
            "id": c.id,
            "requirement_id": c.requirement_id,
            "previous_cluster_id": c.previous_cluster_id,
            "new_cluster_id": c.new_cluster_id,
            "confidence_score": c.confidence_score,
            "comments": c.comments,
            "applied_by": c.applied_by,
            "status": c.status,
            "created_at": c.created_at.isoformat(),
        })
    return json.dumps(data, indent=2)
