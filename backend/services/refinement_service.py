"""Phase 3 refinement service layer.

Connects ML suggestion algorithms and LLM rationale generation to the
database and public API. Provides:
- generate_and_persist_suggestions: full suggestion pipeline
- apply_suggestion: accept/reject and mutate cluster assignments
- get_suggestions: list persisted suggestions
- get_audit_log: retrieve audit trail
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import numpy as np
from sqlalchemy.orm import Session as DBSession

from core.embeddings import generate_embeddings
from core.labeling import label_clusters
from core.merge_suggest import suggest_merges
from core.representatives import extract_representatives
from core.split_suggest import suggest_splits
from llm_services.refinement import (
    get_refinement_provider,
    score_all_clusters,
)
from models.database import (
    Cluster,
    RefinementAuditLog,
    RefinementSuggestion,
    Requirement,
    Session,
    utcnow,
)

logger = logging.getLogger(__name__)


class RefinementServiceError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


def _safe(value: object, max_chars: int = 500) -> str:
    text = str(value or "").strip()
    return text[:max_chars]


def _json_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _load_session_and_clusters(
    db: DBSession, session_id: int
) -> tuple[Session, list[Requirement], list[Cluster]]:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise RefinementServiceError(404, f"Session {session_id} not found.")
    if session.status != "done":
        raise RefinementServiceError(
            400, "Clustering must be completed before generating refinement suggestions."
        )

    reqs = (
        db.query(Requirement)
        .filter(Requirement.session_id == session_id)
        .order_by(Requirement.id.asc())
        .all()
    )
    if not reqs:
        raise RefinementServiceError(400, "No requirements found for session.")

    clusters = (
        db.query(Cluster)
        .filter(Cluster.session_id == session_id)
        .order_by(Cluster.cluster_id.asc())
        .all()
    )
    if not clusters:
        raise RefinementServiceError(400, "No clusters found. Run clustering first.")

    return session, reqs, clusters


def _rebuild_arrays(
    reqs: list[Requirement],
) -> tuple[list[str], list[str], np.ndarray]:
    """Extract texts, req_ids, and labels from persisted requirements."""
    texts = [r.text for r in reqs]
    req_ids = [r.req_id or f"REQ-{i + 1:03d}" for i, r in enumerate(reqs)]
    labels = np.array(
        [r.cluster_id if r.cluster_id is not None else -1 for r in reqs],
        dtype=np.int32,
    )
    return texts, req_ids, labels


def _suggestion_to_out(s: RefinementSuggestion) -> dict:
    return {
        "id": s.id,
        "session_id": s.session_id,
        "suggestion_type": s.suggestion_type,
        "status": s.status,
        "cluster_a_id": s.cluster_a_id,
        "cluster_b_id": s.cluster_b_id,
        "cluster_id": s.cluster_id,
        "cluster_a_label": s.cluster_a_label,
        "cluster_b_label": s.cluster_b_label,
        "cluster_label": s.cluster_label,
        "similarity_score": s.similarity_score,
        "silhouette_delta": s.silhouette_delta,
        "coherence_score": s.coherence_score,
        "spread_score": s.spread_score,
        "bimodality_score": s.bimodality_score,
        "rationale": s.rationale,
        "summary": s.summary,
        "representative_req_ids": _json_list(s.representative_req_ids_json),
        "sub_cluster_sizes": _json_list(s.sub_cluster_sizes_json),
        "created_at": s.created_at,
    }


def generate_and_persist_suggestions(
    db: DBSession,
    request: Any,
) -> dict:
    """Full suggestion pipeline: analyze clusters and persist suggestions.

    1. Load session, clusters, requirements.
    2. Re-generate embeddings (from cache) and rebuild labels.
    3. Run merge and split suggestion algorithms.
    4. Score coherence and generate rationales via provider.
    5. Extract representatives.
    6. Persist suggestions to DB.
    """
    session_id = int(request.session_id)
    session, reqs, clusters = _load_session_and_clusters(db, session_id)
    texts, req_ids, labels = _rebuild_arrays(reqs)

    warnings: list[str] = []

    # Step 1: Re-generate embeddings from cache
    try:
        embeddings = generate_embeddings(texts, batch_size=64, use_cache=True)
    except Exception as exc:
        raise RefinementServiceError(
            500, f"Failed to generate embeddings: {_safe(exc)}"
        ) from exc

    # Build 10D embeddings for silhouette analysis
    from core.reduction import reduce_embeddings

    try:
        embeddings_10d, _embeddings_2d = reduce_embeddings(embeddings)
    except Exception as exc:
        raise RefinementServiceError(
            500, f"Failed to reduce embeddings: {_safe(exc)}"
        ) from exc

    # Build cluster info from DB
    cluster_info: Dict[int, Dict] = {}
    cluster_labels_map: Dict[int, str] = {}
    for c in clusters:
        kw = c.keywords if isinstance(c.keywords, list) else []
        cluster_info[c.cluster_id] = {
            "label": c.label,
            "keywords": kw,
            "size": c.size,
        }
        cluster_labels_map[c.cluster_id] = c.label

    # Step 2: Run merge suggestions
    merge_suggestions = suggest_merges(
        embeddings,
        embeddings_10d,
        labels,
        top_n=request.top_n_merges,
        sim_threshold=request.sim_threshold,
    )

    # Step 3: Run split suggestions
    split_suggestions = suggest_splits(
        embeddings,
        embeddings_10d,
        labels,
        top_n=request.top_n_splits,
        spread_threshold=request.spread_threshold,
    )

    # Step 4: Score coherence
    provider = get_refinement_provider(request.provider_name)
    coherence_results = score_all_clusters(
        embeddings, texts, labels, cluster_info, request.provider_name
    )

    # Step 5: Extract representatives
    reps_by_cluster = extract_representatives(embeddings, texts, labels, req_ids, top_n=3)

    # Step 6: Generate rationales and summaries
    # Clear old suggestions for this session
    db.query(RefinementSuggestion).filter(
        RefinementSuggestion.session_id == session_id,
        RefinementSuggestion.status == "pending",
    ).delete()

    persisted_merge: list[dict] = []
    for ms in merge_suggestions:
        label_a = cluster_labels_map.get(ms.cluster_a, f"Cluster {ms.cluster_a}")
        label_b = cluster_labels_map.get(ms.cluster_b, f"Cluster {ms.cluster_b}")
        coh_a = coherence_results.get(ms.cluster_a)
        coh_b = coherence_results.get(ms.cluster_b)

        rationale = provider.generate_merge_rationale(
            label_a,
            label_b,
            ms.centroid_similarity,
            ms.silhouette_delta,
            coh_a.coherence_score if coh_a else 0.0,
            coh_b.coherence_score if coh_b else 0.0,
        )

        rep_ids_a = [r.req_id for r in reps_by_cluster.get(ms.cluster_a, [])]
        rep_ids_b = [r.req_id for r in reps_by_cluster.get(ms.cluster_b, [])]
        rep_texts = [
            r.text for r in reps_by_cluster.get(ms.cluster_a, [])
        ] + [
            r.text for r in reps_by_cluster.get(ms.cluster_b, [])
        ]

        summary = provider.generate_cluster_summary(
            f"{label_a} + {label_b}",
            (cluster_info.get(ms.cluster_a, {}).get("keywords", [])
             + cluster_info.get(ms.cluster_b, {}).get("keywords", [])),
            rep_texts[:3],
        )

        row = RefinementSuggestion(
            session_id=session_id,
            suggestion_type="merge",
            status="pending",
            cluster_a_id=ms.cluster_a,
            cluster_b_id=ms.cluster_b,
            cluster_a_label=label_a,
            cluster_b_label=label_b,
            similarity_score=ms.centroid_similarity,
            silhouette_delta=ms.silhouette_delta,
            coherence_score=(
                (ms.coherence_a + ms.coherence_b) / 2 if ms.coherence_a and ms.coherence_b else None
            ),
            rationale=rationale,
            summary=summary,
            representative_req_ids_json=rep_ids_a + rep_ids_b,
            metadata_json=ms.to_dict(),
        )
        db.add(row)
        db.flush()
        persisted_merge.append(_suggestion_to_out(row))

    persisted_split: list[dict] = []
    for ss in split_suggestions:
        label = cluster_labels_map.get(ss.cluster_id, f"Cluster {ss.cluster_id}")

        rationale = provider.generate_split_rationale(
            label,
            ss.bimodality_score,
            ss.bic_improvement,
            ss.spread_score,
            ss.sub_cluster_sizes,
        )

        rep_ids = [r.req_id for r in reps_by_cluster.get(ss.cluster_id, [])]
        rep_texts = [r.text for r in reps_by_cluster.get(ss.cluster_id, [])]

        summary = provider.generate_cluster_summary(
            label,
            cluster_info.get(ss.cluster_id, {}).get("keywords", []),
            rep_texts[:3],
        )

        row = RefinementSuggestion(
            session_id=session_id,
            suggestion_type="split",
            status="pending",
            cluster_id=ss.cluster_id,
            cluster_label=label,
            spread_score=ss.spread_score,
            bimodality_score=ss.bimodality_score,
            silhouette_delta=ss.silhouette_delta,
            rationale=rationale,
            summary=summary,
            representative_req_ids_json=rep_ids,
            sub_labels_json=ss.sub_labels,
            sub_cluster_sizes_json=ss.sub_cluster_sizes,
            metadata_json=ss.to_dict(),
        )
        db.add(row)
        db.flush()
        persisted_split.append(_suggestion_to_out(row))

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise RefinementServiceError(500, "Failed to persist refinement suggestions.")

    # Build response
    coherence_out = [
        {
            "cluster_id": cr.cluster_id,
            "coherence_score": cr.coherence_score,
            "top_keywords": cr.top_keywords,
            "size": cr.size,
            "assessment": cr.assessment,
        }
        for cr in coherence_results.values()
    ]

    summaries_out = []
    for cid, info in cluster_info.items():
        rep_texts = [r.text for r in reps_by_cluster.get(cid, [])]
        summary = provider.generate_cluster_summary(
            info["label"], info.get("keywords", []), rep_texts[:3]
        )
        summaries_out.append({
            "cluster_id": cid,
            "summary": summary,
            "representative_count": len(reps_by_cluster.get(cid, [])),
        })

    return {
        "session_id": session_id,
        "merge_suggestions": persisted_merge,
        "split_suggestions": persisted_split,
        "coherence_scores": coherence_out,
        "cluster_summaries": summaries_out,
        "warnings": warnings,
    }


def apply_suggestion(db: DBSession, request: Any) -> dict:
    """Accept or reject a refinement suggestion.

    For accept:
    - Merge: reassign requirements from cluster B to cluster A, re-label, update DB.
    - Split: reassign requirements per sub_labels, create new cluster, re-label both.
    For reject: update status only, no data mutation.
    """
    session_id = int(request.session_id)
    suggestion_id = int(request.suggestion_id)
    action = request.action

    suggestion = (
        db.query(RefinementSuggestion)
        .filter(
            RefinementSuggestion.id == suggestion_id,
            RefinementSuggestion.session_id == session_id,
        )
        .first()
    )
    if not suggestion:
        raise RefinementServiceError(404, "Suggestion not found.")
    if suggestion.status != "pending":
        raise RefinementServiceError(400, f"Suggestion is already {suggestion.status}.")

    if action == "reject":
        suggestion.status = "rejected"
        suggestion.updated_at = utcnow()

        audit = RefinementAuditLog(
            session_id=session_id,
            suggestion_id=suggestion_id,
            action="rejected",
        )
        db.add(audit)
        db.commit()

        return {
            "suggestion_id": suggestion_id,
            "action": "reject",
            "status": "rejected",
            "affected_clusters": [],
            "message": "Suggestion rejected.",
        }

    # Accept and apply
    session, reqs, clusters = _load_session_and_clusters(db, session_id)
    texts, req_ids, labels = _rebuild_arrays(reqs)

    # Capture before state
    before_state = {
        "clusters": [
            {"cluster_id": c.cluster_id, "label": c.label, "size": c.size}
            for c in clusters
        ],
        "requirement_assignments": {
            str(r.id): r.cluster_id for r in reqs
        },
    }

    affected_clusters: list[int] = []

    if suggestion.suggestion_type == "merge":
        cluster_a = suggestion.cluster_a_id
        cluster_b = suggestion.cluster_b_id
        if cluster_a is None or cluster_b is None:
            raise RefinementServiceError(400, "Invalid merge suggestion: missing cluster IDs.")

        # Reassign requirements from cluster B to cluster A
        db.query(Requirement).filter(
            Requirement.session_id == session_id,
            Requirement.cluster_id == cluster_b,
        ).update({"cluster_id": cluster_a})

        # Delete old cluster B
        db.query(Cluster).filter(
            Cluster.session_id == session_id,
            Cluster.cluster_id == cluster_b,
        ).delete()

        # Re-label merged cluster
        merged_reqs = (
            db.query(Requirement)
            .filter(
                Requirement.session_id == session_id,
                Requirement.cluster_id == cluster_a,
            )
            .all()
        )
        merged_texts = [r.text for r in merged_reqs]
        merged_labels = np.array([cluster_a] * len(merged_texts))
        new_info = label_clusters(merged_texts, merged_labels)

        cluster_a_row = (
            db.query(Cluster)
            .filter(Cluster.session_id == session_id, Cluster.cluster_id == cluster_a)
            .first()
        )
        if cluster_a_row and cluster_a in new_info:
            cluster_a_row.label = new_info[cluster_a]["label"]
            cluster_a_row.keywords = new_info[cluster_a]["keywords"]
            cluster_a_row.size = len(merged_texts)

        affected_clusters = [cluster_a, cluster_b]

    elif suggestion.suggestion_type == "split":
        cluster_id = suggestion.cluster_id
        if cluster_id is None:
            raise RefinementServiceError(400, "Invalid split suggestion: missing cluster ID.")

        sub_labels = _json_list(suggestion.sub_labels_json)
        if not sub_labels:
            raise RefinementServiceError(400, "Invalid split suggestion: missing sub-labels.")

        # Get requirements in this cluster
        cluster_reqs = (
            db.query(Requirement)
            .filter(
                Requirement.session_id == session_id,
                Requirement.cluster_id == cluster_id,
            )
            .order_by(Requirement.id.asc())
            .all()
        )

        if len(cluster_reqs) != len(sub_labels):
            raise RefinementServiceError(
                400,
                f"Sub-label count ({len(sub_labels)}) does not match cluster size ({len(cluster_reqs)}).",
            )

        # Find the next available cluster ID
        max_cluster_id = (
            db.query(Cluster.cluster_id)
            .filter(Cluster.session_id == session_id)
            .order_by(Cluster.cluster_id.desc())
            .first()
        )
        new_cluster_id = (max_cluster_id[0] + 1) if max_cluster_id else 1

        # Reassign sub-group 1 to new cluster
        for i, req in enumerate(cluster_reqs):
            if sub_labels[i] == 1:
                req.cluster_id = new_cluster_id

        # Re-label both clusters
        group_0_reqs = [r for i, r in enumerate(cluster_reqs) if sub_labels[i] == 0]
        group_1_reqs = [r for i, r in enumerate(cluster_reqs) if sub_labels[i] == 1]

        if group_0_reqs:
            texts_0 = [r.text for r in group_0_reqs]
            labels_0 = np.array([cluster_id] * len(texts_0))
            info_0 = label_clusters(texts_0, labels_0)
            old_cluster = (
                db.query(Cluster)
                .filter(Cluster.session_id == session_id, Cluster.cluster_id == cluster_id)
                .first()
            )
            if old_cluster and cluster_id in info_0:
                old_cluster.label = info_0[cluster_id]["label"]
                old_cluster.keywords = info_0[cluster_id]["keywords"]
                old_cluster.size = len(group_0_reqs)

        if group_1_reqs:
            texts_1 = [r.text for r in group_1_reqs]
            labels_1 = np.array([new_cluster_id] * len(texts_1))
            info_1 = label_clusters(texts_1, labels_1)
            new_label_info = info_1.get(new_cluster_id, {"label": "New Cluster", "keywords": [], "size": len(group_1_reqs)})
            new_cluster = Cluster(
                session_id=session_id,
                cluster_id=new_cluster_id,
                label=new_label_info["label"],
                keywords=new_label_info["keywords"],
                size=len(group_1_reqs),
            )
            db.add(new_cluster)

        affected_clusters = [cluster_id, new_cluster_id]

    else:
        raise RefinementServiceError(400, f"Unknown suggestion type: {suggestion.suggestion_type}")

    # Capture after state
    db.flush()
    after_clusters = db.query(Cluster).filter(Cluster.session_id == session_id).all()
    # Recompute the cluster total from the flushed rows (covers both merge and
    # split without relying on autoflush timing).
    session.total_clusters = len(after_clusters)
    after_state = {
        "clusters": [
            {"cluster_id": c.cluster_id, "label": c.label, "size": c.size}
            for c in after_clusters
        ],
        "requirement_assignments": {
            str(r.id): r.cluster_id
            for r in db.query(Requirement).filter(Requirement.session_id == session_id).all()
        },
    }

    suggestion.status = "applied"
    suggestion.applied_at = utcnow()
    suggestion.updated_at = utcnow()

    audit = RefinementAuditLog(
        session_id=session_id,
        suggestion_id=suggestion_id,
        action="applied",
        before_state_json=before_state,
        after_state_json=after_state,
    )
    db.add(audit)

    try:
        db.commit()
    except Exception:
        db.rollback()
        raise RefinementServiceError(500, "Failed to apply suggestion.")

    return {
        "suggestion_id": suggestion_id,
        "action": "accept",
        "status": "applied",
        "affected_clusters": affected_clusters,
        "message": f"{'Merge' if suggestion.suggestion_type == 'merge' else 'Split'} applied successfully.",
    }


def get_suggestions(
    db: DBSession,
    session_id: int,
    status_filter: Optional[str] = None,
) -> list[dict]:
    """List persisted refinement suggestions for a session."""
    query = db.query(RefinementSuggestion).filter(
        RefinementSuggestion.session_id == session_id
    )
    if status_filter:
        query = query.filter(RefinementSuggestion.status == status_filter)
    rows = query.order_by(RefinementSuggestion.created_at.desc()).all()
    return [_suggestion_to_out(row) for row in rows]


def get_audit_log(db: DBSession, session_id: int) -> list[dict]:
    """Retrieve audit trail of applied refinements for a session."""
    rows = (
        db.query(RefinementAuditLog)
        .filter(RefinementAuditLog.session_id == session_id)
        .order_by(RefinementAuditLog.created_at.desc())
        .all()
    )
    return [
        {
            "id": row.id,
            "session_id": row.session_id,
            "suggestion_id": row.suggestion_id,
            "action": row.action,
            "before_state": row.before_state_json,
            "after_state": row.after_state_json,
            "applied_by": row.applied_by,
            "created_at": row.created_at,
        }
        for row in rows
    ]
