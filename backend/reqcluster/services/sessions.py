"""Session read models and queries. Every function opens its own short DB scope."""
import contextlib
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import delete as sa_delete, func, insert, select

from reqcluster.config import get_settings
from reqcluster.db import session_scope
from reqcluster.domain import ClusterInfo, Job, SessionStatus
from reqcluster.errors import Conflict, NotFound
from reqcluster.ingest import IngestResult
from reqcluster.jobs import get_runner
from reqcluster.models import (
    AssignmentRow,
    ClusterRow,
    GraphRow,
    JobRow,
    RequirementRow,
    RunRow,
    SessionRow,
)


class SessionOut(BaseModel):
    id: int
    name: str
    created_at: datetime
    total_requirements: int
    status: SessionStatus
    total_clusters: int
    noise_count: int
    revision: int
    last_job: Job | None


class RequirementOut(BaseModel):
    id: int
    req_id: str
    ordinal: int
    text: str
    cluster_id: int | None
    is_noise: bool | None
    probability: float | None
    x: float | None
    y: float | None
    attributes: dict[str, str] = {}


class ClusterOut(ClusterInfo):
    pass


class ClusterDetail(ClusterOut):
    requirements: list[RequirementOut]


class GraphNode(BaseModel):
    id: int
    req_id: str
    cluster_id: int | None
    is_noise: bool | None
    x: float | None
    y: float | None
    text: str


class GraphEdge(BaseModel):
    source: int
    target: int
    weight: float


class GraphOut(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class MetricsOut(BaseModel):
    revision: int
    n_clusters: int
    noise_count: int
    metrics: dict
    warnings: list[str]
    durations: dict
    embedding_comparison: dict | None = None
    ablation_report: dict | None = None


def _row(s, sid: int) -> SessionRow:
    row = s.get(SessionRow, sid)
    if row is None:
        raise NotFound(f"session {sid} not found")
    return row


def _current_run(s, sid: int) -> RunRow:
    run_id = _row(s, sid).current_run_id
    run = s.get(RunRow, run_id) if run_id is not None else None
    if run is None:
        raise Conflict(f"session {sid} has not been clustered yet")
    return run


def _out(s, row: SessionRow) -> SessionOut:
    run = s.get(RunRow, row.current_run_id) if row.current_run_id is not None else None
    job = get_runner().latest(row.id, ("cluster", "recluster"))
    if job is not None and job.status in ("queued", "running"):
        status = job.status
    elif job is not None and job.status in ("failed", "cancelled"):
        status = "error"
    else:
        status = "done" if run else "uploaded"
    return SessionOut(
        id=row.id, name=row.name, created_at=row.created_at,
        total_requirements=row.n_requirements, status=status,
        total_clusters=run.n_clusters if run else 0,
        noise_count=run.noise_count if run else 0,
        revision=run.revision if run else 0, last_job=job,
    )


def create(ingest_result: IngestResult, name: str) -> SessionOut:
    reqs = ingest_result.requirements
    with session_scope() as s:
        row = SessionRow(name=name, n_requirements=len(reqs))
        s.add(row)
        s.flush()
        if reqs:
            s.execute(insert(RequirementRow), [
                {"session_id": row.id, "ordinal": r.ordinal, "req_id": r.req_id, "text": r.text,
                 "attributes": r.attributes}
                for r in reqs
            ])
        return _out(s, row)


def list_sessions() -> list[SessionOut]:
    with session_scope() as s:
        rows = s.scalars(select(SessionRow).order_by(SessionRow.id.desc())).all()
        return [_out(s, r) for r in rows]


def get(sid: int) -> SessionOut:
    with session_scope() as s:
        return _out(s, _row(s, sid))


def unlink_runs(run_ids) -> None:
    d = get_settings().runs_dir
    for rid in run_ids:
        with contextlib.suppress(FileNotFoundError):
            (d / f"{rid}.npz").unlink()



def delete(sid: int) -> None:
    with session_scope() as s:
        _row(s, sid)
        active = s.scalar(
            select(func.count()).select_from(JobRow).where(
                JobRow.session_id == sid, JobRow.status.in_(("queued", "running")))
        )
        if active:
            raise Conflict(f"session {sid} has an active job")
        run_ids = list(s.scalars(select(RunRow.id).where(RunRow.session_id == sid)))
        s.execute(sa_delete(SessionRow).where(SessionRow.id == sid))
    unlink_runs(run_ids)


def clusters(sid: int) -> list[ClusterOut]:
    with session_scope() as s:
        run = _current_run(s, sid)
        rows = s.scalars(select(ClusterRow).where(ClusterRow.run_id == run.id)
                         .order_by(ClusterRow.cluster_id)).all()
        return [ClusterOut(cluster_id=c.cluster_id, size=c.size, label=c.label,
                           keywords=c.keywords) for c in rows]


def _req_query(run_id):
    a = AssignmentRow
    j = (a.requirement_id == RequirementRow.id) & (a.run_id == (run_id if run_id is not None else -1))
    return (
        select(RequirementRow, a.cluster_id, a.probability, a.x, a.y)
        .select_from(RequirementRow).outerjoin(a, j)
    )


def _req_out(r, cid, prob, x, y) -> RequirementOut:
    return RequirementOut(
        id=r.id, req_id=r.req_id, ordinal=r.ordinal, text=r.text, cluster_id=cid,
        is_noise=None if cid is None else cid == -1, probability=prob, x=x, y=y,
        attributes=r.attributes or {})


def cluster_detail(sid: int, cid: int) -> ClusterDetail:
    with session_scope() as s:
        run = _current_run(s, sid)
        c = s.get(ClusterRow, (run.id, cid))
        if c is None:
            raise NotFound(f"cluster {cid} not found in session {sid}")
        q = (_req_query(run.id).where(RequirementRow.session_id == sid,
                                      AssignmentRow.cluster_id == cid)
             .order_by(RequirementRow.ordinal))
        reqs = [_req_out(*t) for t in s.execute(q)]
        return ClusterDetail(cluster_id=cid, size=c.size, label=c.label,
                             keywords=c.keywords, requirements=reqs)


def graph(sid: int) -> GraphOut:
    with session_scope() as s:
        run = _current_run(s, sid)
        q = _req_query(run.id).where(RequirementRow.session_id == sid).order_by(
            RequirementRow.ordinal)
        nodes = []
        for r, cid, _p, x, y in s.execute(q):
            t = r.text if len(r.text) <= 200 else r.text[:200] + "…"
            nodes.append(GraphNode(id=r.id, req_id=r.req_id, cluster_id=cid,
                                   is_noise=None if cid is None else cid == -1, x=x, y=y, text=t))
        g = s.get(GraphRow, run.id)
        edges = [GraphEdge(source=a, target=b, weight=w) for a, b, w in (g.edges if g else [])]
        return GraphOut(nodes=nodes, edges=edges)


def metrics(sid: int) -> MetricsOut:
    with session_scope() as s:
        run = _current_run(s, sid)
        return MetricsOut(revision=run.revision, n_clusters=run.n_clusters,
                          noise_count=run.noise_count, metrics=run.metrics,
                          warnings=run.warnings, durations=run.durations,
                          embedding_comparison=(run.reports or {}).get("embedding_comparison"),
                          ablation_report=(run.reports or {}).get("ablation_report"))


_SORT = {"ordinal": RequirementRow.ordinal, "req_id": RequirementRow.req_id,
         "text": RequirementRow.text, "cluster_id": AssignmentRow.cluster_id,
         "probability": AssignmentRow.probability}


def _like_escape(v: str) -> str:
    return v.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def requirements(sid: int, cluster_id: int | None = None, noise: bool | None = None,
                 search: str | None = None, sort: str = "ordinal", dir: str = "asc",
                 page: int = 1, page_size: int = 50) -> tuple[list[RequirementOut], int]:
    with session_scope() as s:
        run_id = _row(s, sid).current_run_id
        q = _req_query(run_id).where(RequirementRow.session_id == sid)
        if cluster_id is not None:
            q = q.where(AssignmentRow.cluster_id == cluster_id)
        if noise is not None:
            q = q.where(AssignmentRow.cluster_id == -1 if noise else AssignmentRow.cluster_id != -1)
        if search:
            pat = f"%{_like_escape(search)}%"
            q = q.where(RequirementRow.text.ilike(pat, escape="\\"))
        total = s.scalar(select(func.count()).select_from(q.subquery()))
        col = _SORT.get(sort, RequirementRow.ordinal)
        col = col.desc() if dir == "desc" else col.asc()
        q = q.order_by(col, RequirementRow.ordinal).offset((page - 1) * page_size).limit(page_size)
        return [_req_out(*t) for t in s.execute(q)], total
