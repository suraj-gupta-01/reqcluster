"""Shared write path for edits inside a run (feedback moves, suggestion merge/split)."""
import numpy as np
from sqlalchemy import delete, insert, select, update
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import set_committed_value

from reqcluster.errors import Stale
from reqcluster.ml import keywords
from reqcluster.models import AssignmentRow, AuditRow, ClusterRow, RequirementRow, RunRow


def reassign(s: Session, run: RunRow, moves: dict[int, int], *, action: str, actor: str = "",
             details: dict | None = None) -> None:
    """Apply {requirement_pk: cluster_id} inside the caller's transaction.

    Cluster ids are kept stable (no relabel); emptied clusters disappear. c-TF-IDF labels are
    relative to all clusters, so every cluster row of the run is recomputed, then edit_seq is
    bumped (pending suggestions generated before this edit become stale) and an audit row written.
    """
    # Claim the edit first: a conditional bump makes two concurrent edits of the same revision
    # (two accepts, accept + feedback move) fail cleanly instead of both applying.
    claimed = s.execute(update(RunRow).where(RunRow.id == run.id, RunRow.edit_seq == run.edit_seq)
                        .values(edit_seq=RunRow.edit_seq + 1)).rowcount
    if not claimed:
        raise Stale("clusters were changed by another edit; refresh and retry")
    set_committed_value(run, "edit_seq", run.edit_seq + 1)
    by_target: dict[int, list[int]] = {}
    for req_id, cid in moves.items():
        by_target.setdefault(int(cid), []).append(int(req_id))
    for cid, req_ids in by_target.items():
        s.execute(update(AssignmentRow)
                  .where(AssignmentRow.run_id == run.id, AssignmentRow.requirement_id.in_(req_ids))
                  .values(cluster_id=cid, probability=1.0))  # a human decision is certain

    rows = s.execute(select(AssignmentRow.cluster_id, RequirementRow.text)
                     .join(RequirementRow, RequirementRow.id == AssignmentRow.requirement_id)
                     .where(AssignmentRow.run_id == run.id)
                     .order_by(RequirementRow.ordinal)).all()
    labels = np.array([r[0] for r in rows], dtype=np.int64)
    infos = keywords.label_clusters([r[1] for r in rows], labels)
    s.execute(delete(ClusterRow).where(ClusterRow.run_id == run.id))
    if infos:
        s.execute(insert(ClusterRow), [
            {"run_id": run.id, "cluster_id": c.cluster_id, "size": c.size, "label": c.label,
             "keywords": c.keywords} for c in infos])
    run.n_clusters = len(infos)
    run.noise_count = int((labels == -1).sum())
    n = len(labels)
    rate = run.noise_count / n if n else 0.0
    # silhouette is stale after a manual edit; the other counts are recomputed
    run.metrics = {**(run.metrics or {}), "n": n, "n_clusters": run.n_clusters,
                   "noise_count": run.noise_count, "noise_rate": rate, "coverage": 1 - rate,
                   "silhouette": None}
    s.add(AuditRow(run_id=run.id, action=action, actor=actor,
                   details={"moves": {str(k): v for k, v in moves.items()}, **(details or {})}))
