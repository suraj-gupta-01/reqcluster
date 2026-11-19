"""Human corrections: moves, review, derived must/cannot constraints, constrained re-cluster."""
import csv
import io
import re
from datetime import datetime, UTC

import numpy as np
from pydantic import BaseModel
from sqlalchemy import delete, insert, select

from reqcluster.config import get_settings
from reqcluster.db import session_scope
from reqcluster.domain import Job
from reqcluster.errors import Conflict, NotFound
from reqcluster.export.bundle import _guard
from reqcluster.jobs import JobContext, get_runner
from reqcluster.ml import constraints as C
from reqcluster.ml import keywords, metrics
from reqcluster.models import (
    AssignmentRow,
    ClusterRow,
    ConstraintRow,
    FeedbackRow,
    RequirementRow,
    SessionRow,
)
from reqcluster.services.assignments import reassign
from reqcluster.services.clustering import (
    ClusteringSummary,
    load_run_arrays,
    next_cluster_id,
    persist_run,
    run_embeddings,
)
from reqcluster.services.sessions import _current_run


class ReclusterSummary(ClusteringSummary):
    moved: int


class SubmitOut(BaseModel):
    feedback_id: int | None  # None for a no-op move into the current cluster
    requirement_id: int
    old_cluster_id: int
    new_cluster_id: int
    status: str
    constraints_added: int


class FeedbackOut(BaseModel):
    id: int
    requirement_id: int
    req_id: str
    text: str
    old_cluster_id: int
    new_cluster_id: int
    status: str
    comments: str
    applied_by: str
    created_at: datetime
    reviewed_at: datetime | None
    confidence: float


class ConstraintPair(BaseModel):
    a: int
    b: int
    kind: str
    feedback_id: int | None


class ConstraintsOut(BaseModel):
    constraint_pairs: list[ConstraintPair]
    conflicts: list[dict]


def confidence(comment: str, label: str, kws: list[str]) -> float:
    """Keyword overlap of the comment with the target cluster, in [0.5, 1.0]."""
    if not comment.strip():
        return 1.0
    text = comment.lower()
    hits = 0
    for kw in (k.lower() for k in kws):
        if kw in text:
            hits += 2
        else:
            hits += sum(1 for w in re.findall(r"\w+", kw) if len(w) > 3 and w in text)
    hits += 2 * sum(1 for w in re.findall(r"\w+", label.lower()) if len(w) > 3 and w in text)
    return round(0.5 + min(hits * 0.1, 0.5), 2)


def _ordered(s, run_id: int, sid: int):
    """(requirement pks, texts, labels) in ordinal order."""
    rows = s.execute(select(RequirementRow.id, RequirementRow.text, AssignmentRow.cluster_id)
                     .join(AssignmentRow, (AssignmentRow.requirement_id == RequirementRow.id)
                           & (AssignmentRow.run_id == run_id))
                     .where(RequirementRow.session_id == sid)
                     .order_by(RequirementRow.ordinal)).all()
    return ([r[0] for r in rows], [r[1] for r in rows],
            np.array([r[2] for r in rows], dtype=np.int64))


def _check_idle(sid: int) -> None:
    job = get_runner().latest(sid, ("cluster", "recluster"))
    if job is not None and job.status in ("queued", "running"):
        raise Conflict(f"a {job.kind} job is active for session {sid}")


def submit(sid: int, requirement_id: int, new_cluster_id: int | None, comments: str = "",
           applied_by: str = "") -> SubmitOut:
    new = -1 if new_cluster_id is None else int(new_cluster_id)
    with session_scope() as s:
        if s.get(SessionRow, sid) is None:
            raise NotFound(f"session {sid} not found")
        req = s.get(RequirementRow, requirement_id)
        if req is None or req.session_id != sid:
            raise NotFound(f"requirement {requirement_id} not found")
    _check_idle(sid)
    with session_scope() as s:
        run = _current_run(s, sid)
        if new != -1 and s.get(ClusterRow, (run.id, new)) is None:
            raise NotFound(f"cluster {new} not found")
        old = s.get(AssignmentRow, (run.id, requirement_id)).cluster_id
        if old == new:
            return SubmitOut(feedback_id=None, requirement_id=requirement_id, old_cluster_id=old,
                             new_cluster_id=new, status="noop", constraints_added=0)
        reassign(s, run, {requirement_id: new}, action="feedback_move", actor=applied_by,
                 details={"old": old, "new": new})
        fb = FeedbackRow(run_id=run.id, requirement_id=requirement_id, old_cluster_id=old,
                         new_cluster_id=new, comments=comments or "", applied_by=applied_by or "")
        s.add(fb)
        s.flush()
        pks, _, labels = _ordered(s, run.id, sid)
        added = 0
        npz = get_settings().runs_dir / f"{run.id}.npz"
        if npz.exists():
            x10 = load_run_arrays(run.id)["x10"]
            must, cannot = C.from_correction(labels, pks.index(requirement_id), new, old, X=x10)
            want = {(min(pks[i], pks[j]), max(pks[i], pks[j]), kind)
                    for kind, pairs in (("must", must), ("cannot", cannot)) for i, j in pairs}
            have = set(s.execute(select(ConstraintRow.a_id, ConstraintRow.b_id, ConstraintRow.kind)
                                 .where(ConstraintRow.run_id == run.id)).all())
            fresh = sorted(want - {tuple(h) for h in have})
            if fresh:
                s.execute(insert(ConstraintRow), [
                    {"run_id": run.id, "a_id": a, "b_id": b, "kind": k, "feedback_id": fb.id}
                    for a, b, k in fresh])
            added = len(fresh)
        return SubmitOut(feedback_id=fb.id, requirement_id=requirement_id, old_cluster_id=old,
                         new_cluster_id=new, status=fb.status, constraints_added=added)


def _out(s, fb: FeedbackRow, req_id: str, text: str) -> FeedbackOut:
    c = s.get(ClusterRow, (fb.run_id, fb.new_cluster_id))
    conf = confidence(fb.comments, c.label, c.keywords) if c else 1.0
    return FeedbackOut(id=fb.id, requirement_id=fb.requirement_id, req_id=req_id, text=text,
                       old_cluster_id=fb.old_cluster_id, new_cluster_id=fb.new_cluster_id,
                       status=fb.status, comments=fb.comments, applied_by=fb.applied_by,
                       created_at=fb.created_at, reviewed_at=fb.reviewed_at, confidence=conf)


def review(sid: int, feedback_id: int, approve: bool) -> FeedbackOut:
    _check_idle(sid)
    with session_scope() as s:
        run = _current_run(s, sid)
        fb = s.get(FeedbackRow, feedback_id)
        if fb is None or fb.run_id != run.id:
            raise NotFound(f"feedback {feedback_id} not found")
        if fb.status != "pending":
            raise Conflict(f"feedback {feedback_id} is already {fb.status}")
        fb.status = "approved" if approve else "rejected"
        fb.reviewed_at = datetime.now(UTC)
        if not approve:
            a = s.get(AssignmentRow, (run.id, fb.requirement_id))
            if a.cluster_id == fb.new_cluster_id:
                back = fb.old_cluster_id
                if back != -1 and s.get(ClusterRow, (run.id, back)) is None:
                    back = next_cluster_id(s, run)  # old cluster was merged away: never revive its id
                reassign(s, run, {fb.requirement_id: back}, action="feedback_reject",
                         actor=fb.applied_by, details={"feedback_id": fb.id})
            s.execute(delete(ConstraintRow).where(ConstraintRow.feedback_id == fb.id))
        r = s.get(RequirementRow, fb.requirement_id)
        return _out(s, fb, r.req_id, r.text)


def _rows(sid: int, status: str | None):
    with session_scope() as s:
        run = _current_run(s, sid)
        q = (select(FeedbackRow, RequirementRow.req_id, RequirementRow.text)
             .join(RequirementRow, RequirementRow.id == FeedbackRow.requirement_id)
             .where(FeedbackRow.run_id == run.id).order_by(FeedbackRow.id))
        if status:
            q = q.where(FeedbackRow.status == status)
        return [_out(s, fb, rid, txt) for fb, rid, txt in s.execute(q)]


def queue(sid: int, status: str | None = None) -> list[FeedbackOut]:
    return _rows(sid, status)


def constraints(sid: int) -> ConstraintsOut:
    with session_scope() as s:
        run = _current_run(s, sid)
        rows = s.scalars(select(ConstraintRow).where(ConstraintRow.run_id == run.id)
                         .order_by(ConstraintRow.id)).all()
        pairs = [ConstraintPair(a=r.a_id, b=r.b_id, kind=r.kind, feedback_id=r.feedback_id)
                 for r in rows]
    kinds: dict[tuple[int, int], set[str]] = {}
    for p in pairs:
        kinds.setdefault((p.a, p.b), set()).add(p.kind)
    # shortcut: only direct must/cannot clashes, not transitive ones (apply() still rejects those)
    conflicts = [{"a": a, "b": b, "kinds": sorted(k)} for (a, b), k in kinds.items() if len(k) > 1]
    if conflicts:
        ids = {i for c in conflicts for i in (c["a"], c["b"])}
        with session_scope() as s:
            rid = dict(s.execute(select(RequirementRow.id, RequirementRow.req_id)
                                 .where(RequirementRow.id.in_(ids))))
        for c in conflicts:
            c["a_req_id"], c["b_req_id"] = rid.get(c["a"]), rid.get(c["b"])
    return ConstraintsOut(constraint_pairs=pairs, conflicts=conflicts)


def export(sid: int, fmt: str) -> tuple[bytes, str]:
    items = _rows(sid, None)
    if fmt == "json":
        return ("[" + ",".join(i.model_dump_json() for i in items) + "]").encode(), "application/json"
    cols = ["id", "requirement_id", "req_id", "old_cluster_id", "new_cluster_id", "status",
            "comments", "applied_by", "created_at", "reviewed_at", "confidence"]
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(cols)
    for i in items:
        d = i.model_dump()
        w.writerow([_guard(d[c]) for c in cols])
    return buf.getvalue().encode("utf-8"), "text/csv"


def recluster(sid: int) -> Job:
    with session_scope() as s:
        _current_run(s, sid)
    return get_runner().submit("recluster", sid, lambda ctx: _recluster(ctx, sid), {})


def _recluster(ctx: JobContext, sid: int) -> ReclusterSummary:
    with session_scope() as s:
        run = _current_run(s, sid)
        rid, params = run.id, dict(run.params)
        pks, texts, labels = _ordered(s, rid, sid)
        cons = s.execute(select(ConstraintRow.a_id, ConstraintRow.b_id, ConstraintRow.kind,
                                ConstraintRow.feedback_id)
                         .where(ConstraintRow.run_id == rid)).all()
        db_probs = dict(s.execute(select(AssignmentRow.requirement_id, AssignmentRow.probability)
                                  .where(AssignmentRow.run_id == rid)).all())
    npz = load_run_arrays(rid)
    x10, xy = npz["x10"], npz["xy"]
    probs = np.array([1.0 if db_probs[pk] is None else db_probs[pk] for pk in pks])
    pos = {pk: i for i, pk in enumerate(pks)}
    must = [(pos[a], pos[b]) for a, b, k, _ in cons if k == "must"]
    cannot = [(pos[a], pos[b]) for a, b, k, _ in cons if k == "cannot"]

    ctx.progress(0.1, "embedding")
    warnings: list[str] = []
    X = run_embeddings(sid, run, warnings)
    ctx.check_cancel()
    ctx.progress(0.6, "constraints")
    new, moved = C.apply(X, labels, must, cannot)
    seed = params.get("seed")
    m = metrics.compute(new, x10, seed if seed is not None else get_settings().seed)
    clusters = keywords.label_clusters(texts, new)
    ctx.check_cancel()
    new_rid, revision = persist_run(sid, "constrained", params, pks, new, probs, x10, xy,
                                    clusters, npz.get("edges", np.zeros((0, 3))).tolist(), m, warnings, {},
                                    constraints=cons)
    return ReclusterSummary(run_id=new_rid, revision=revision, n_clusters=len(clusters),
                             noise_count=int((new == -1).sum()), warnings=warnings, durations={},
                             metrics=m, moved=int(moved))
