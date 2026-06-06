"""Feedback-to-embedding bridge and constraint generator.

Translates human cluster reassignments into must-link and cannot-link
constraints, and detects logical conflicts in active constraints.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from sqlalchemy.orm import Session as DBSession

from core.embeddings import generate_embeddings
from core.representatives import extract_representatives
from models.database import ConstraintPair, FeedbackCorrection, Requirement


def generate_constraints_for_correction(
    db: DBSession,
    session_id: int,
    feedback_id: int,
    requirement_id: int,
    old_cluster_id: Optional[int],
    new_cluster_id: Optional[int],
) -> List[ConstraintPair]:
    """Generate must-link and cannot-link constraints from a human correction.

    - Must-link constraints are created between the requirement and the top-3
      representative requirements of the new (target) cluster.
    - Cannot-link constraints are created between the requirement and the top-3
      representative requirements of the old (source) cluster.
    """
    reqs = (
        db.query(Requirement)
        .filter(Requirement.session_id == session_id)
        .order_by(Requirement.id.asc())
        .all()
    )
    if not reqs:
        return []

    texts = [r.text for r in reqs]
    req_ids = [r.req_id or f"REQ-{i + 1:03d}" for i, r in enumerate(reqs)]
    # Use labels currently stored in the DB (which have already been moved)
    labels = np.array(
        [r.cluster_id if r.cluster_id is not None else -1 for r in reqs],
        dtype=np.int32,
    )

    try:
        embeddings = generate_embeddings(texts, use_cache=True)
    except Exception:
        embeddings = np.zeros((len(texts), 384), dtype=np.float32)

    # Maps req_id string and list index to requirement DB id
    req_id_to_db_id = {r.req_id: r.id for r in reqs if r.req_id}
    idx_to_db_id = {i: r.id for i, r in enumerate(reqs)}

    # Verify moved requirement exists
    target_req = db.query(Requirement).filter(Requirement.id == requirement_id).first()
    if not target_req:
        return []

    new_constraints: List[ConstraintPair] = []

    # 1. Create Must-Link constraints with target cluster representatives
    if new_cluster_id is not None and new_cluster_id != -1:
        reps = extract_representatives(embeddings, texts, labels, req_ids, top_n=3)
        target_reps = reps.get(new_cluster_id, [])
        for rep in target_reps:
            rep_db_id = req_id_to_db_id.get(rep.req_id, idx_to_db_id.get(rep.index))
            if rep_db_id and rep_db_id != target_req.id:
                cp = ConstraintPair(
                    session_id=session_id,
                    requirement_a_id=target_req.id,
                    requirement_b_id=rep_db_id,
                    constraint_type="must-link",
                    feedback_id=feedback_id,
                )
                db.add(cp)
                new_constraints.append(cp)

    # 2. Create Cannot-Link constraints with source cluster representatives
    if old_cluster_id is not None and old_cluster_id != -1:
        # Restore labels to old state to calculate old representatives correctly
        old_labels = labels.copy()
        target_idx = next((i for i, r in enumerate(reqs) if r.id == target_req.id), None)
        if target_idx is not None:
            old_labels[target_idx] = old_cluster_id
            reps_old = extract_representatives(embeddings, texts, old_labels, req_ids, top_n=3)
            source_reps = reps_old.get(old_cluster_id, [])
            for rep in source_reps:
                rep_db_id = req_id_to_db_id.get(rep.req_id, idx_to_db_id.get(rep.index))
                if rep_db_id and rep_db_id != target_req.id:
                    cp = ConstraintPair(
                        session_id=session_id,
                        requirement_a_id=target_req.id,
                        requirement_b_id=rep_db_id,
                        constraint_type="cannot-link",
                        feedback_id=feedback_id,
                    )
                    db.add(cp)
                    new_constraints.append(cp)

    db.flush()
    return new_constraints


def detect_constraints_conflicts(db: DBSession, session_id: int) -> List[dict]:
    """Find transitive and direct conflicts in active constraints.

    Checks if a must-link path groups requirements together that are also
    connected by a cannot-link constraint.
    """
    constraints = (
        db.query(ConstraintPair)
        .join(FeedbackCorrection, ConstraintPair.feedback_id == FeedbackCorrection.id)
        .filter(
            ConstraintPair.session_id == session_id,
            FeedbackCorrection.status != "rejected",
        )
        .all()
    )

    if not constraints:
        return []

    req_ids = set()
    for c in constraints:
        req_ids.add(c.requirement_a_id)
        req_ids.add(c.requirement_b_id)

    req_map = {
        r.id: r
        for r in db.query(Requirement).filter(Requirement.id.in_(req_ids)).all()
    }

    # Disjoint Set / Union-Find initialization
    parent = {rid: rid for rid in req_ids}

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> bool:
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j
            return True
        return False

    # Union all must-linked requirements
    must_links = [c for c in constraints if c.constraint_type == "must-link"]
    for c in must_links:
        union(c.requirement_a_id, c.requirement_b_id)

    # Check cannot-links for violations
    cannot_links = [c for c in constraints if c.constraint_type == "cannot-link"]
    conflicts: List[dict] = []
    seen_conflicts = set()

    for c in cannot_links:
        a_id = c.requirement_a_id
        b_id = c.requirement_b_id

        if find(a_id) == find(b_id):
            conflict_key = tuple(sorted([a_id, b_id]))
            if conflict_key in seen_conflicts:
                continue
            seen_conflicts.add(conflict_key)

            req_a = req_map.get(a_id)
            req_b = req_map.get(b_id)

            conflicts.append({
                "requirement_a_id": a_id,
                "requirement_a_label": req_a.req_id if req_a else f"REQ-{a_id}",
                "requirement_a_text": req_a.text if req_a else "",
                "requirement_b_id": b_id,
                "requirement_b_label": req_b.req_id if req_b else f"REQ-{b_id}",
                "requirement_b_text": req_b.text if req_b else "",
                "message": (
                    f"Requirements {req_a.req_id if req_a else a_id} and {req_b.req_id if req_b else b_id} "
                    f"have a cannot-link constraint but are transitively grouped together by must-link corrections."
                )
            })

    return conflicts
