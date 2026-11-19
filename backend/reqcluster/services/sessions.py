"""Session read models and queries. Every function opens its own short DB scope."""
import contextlib
import functools
from datetime import datetime

import numpy as np
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete, func, insert, select, or_

from reqcluster.config import get_settings
from reqcluster.db import session_scope
from reqcluster.domain import ClusterInfo, Job, SessionStatus
from reqcluster.errors import Conflict, NotFound
from reqcluster.ingest import IngestResult
from reqcluster.jobs import get_runner
from reqcluster.models import (
    AssignmentRow,
    ClusterRow,
    JobRow,
    RequirementRow,
    RunRow,
    SessionRow,
    to_job,
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


class GraphNodes(BaseModel):
    id: list[int]
    req_id: list[str]
    cluster_id: list[int | None]
    x: list[float | None]
    y: list[float | None]
    probability: list[float | None]
    label: list[str]  # first LABEL_CHARS of the text, for hover; full text is fetched on demand


class GraphEdges(BaseModel):
    source: list[int]
    target: list[int]
    weight: list[float]


class GraphOut(BaseModel):
    revision: int
    edit_seq: int
    nodes: GraphNodes
    edges: GraphEdges


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


_CLUSTER_KINDS = ("cluster", "recluster")


def _out(s, row: SessionRow, run=None, job=None, preloaded: bool = False) -> SessionOut:
    if not preloaded:
        run = s.get(RunRow, row.current_run_id) if row.current_run_id is not None else None
        job = get_runner().latest(row.id, _CLUSTER_KINDS)
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
        runs = {r.id: r for r in s.scalars(select(RunRow).where(
            RunRow.id.in_([r.current_run_id for r in rows if r.current_run_id is not None])))}
        latest: dict[int, JobRow] = {}  # newest first, so the first row seen per session wins
        for j in s.scalars(select(JobRow).where(JobRow.kind.in_(_CLUSTER_KINDS))
                           .order_by(JobRow.created_at.desc())):
            latest.setdefault(j.session_id, j)
        return [_out(s, r, runs.get(r.current_run_id),
                     to_job(latest[r.id]) if r.id in latest else None, preloaded=True)
                for r in rows]


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


LABEL_CHARS = 24


def _round3(values) -> list:
    """Round a column to 3 decimals in numpy (portable across SQLite/Postgres), keeping None."""
    arr = np.array([np.nan if v is None else v for v in values], dtype=np.float64)
    return [None if np.isnan(v) else v for v in np.round(arr, 3).tolist()]


@functools.lru_cache(maxsize=8)
def _graph(db_url: str, sid: int, run_id: int, edit_seq: int) -> GraphOut:
    """Columnar, text-free payload built from raw rows; cached until the run is edited."""
    a = AssignmentRow
    with session_scope() as s:
        run = s.get(RunRow, run_id)
        q = (select(RequirementRow.id, RequirementRow.req_id,
                    func.substr(RequirementRow.text, 1, LABEL_CHARS),
                    a.cluster_id, a.x, a.y, a.probability)  # rounded below: Postgres lacks round(float, int)
             .select_from(RequirementRow)
             .outerjoin(a, (a.requirement_id == RequirementRow.id) & (a.run_id == run_id))
             .where(RequirementRow.session_id == sid).order_by(RequirementRow.ordinal))
        cols = list(zip(*s.execute(q).all(), strict=False)) or [()] * 7
        cols[4:7] = [_round3(c) for c in cols[4:7]]
        from reqcluster.services.clustering import load_run_arrays  # (clustering imports sessions)
        e = load_run_arrays(run_id).get("edges", np.zeros((0, 3)))
        return GraphOut.model_construct(
            revision=run.revision, edit_seq=edit_seq,
            nodes=GraphNodes.model_construct(**dict(zip(
                ("id", "req_id", "label", "cluster_id", "x", "y", "probability"), map(list, cols), strict=False))),
            edges=GraphEdges.model_construct(  # weights were rounded when the graph was built
                source=e[:, 0].astype(np.int64).tolist(), target=e[:, 1].astype(np.int64).tolist(),
                weight=e[:, 2].tolist()))


def _graph_key(sid: int) -> tuple:
    with session_scope() as s:
        run = _current_run(s, sid)
        return get_settings().database_url, sid, run.id, run.edit_seq


def graph(sid: int) -> GraphOut:
    return _graph(*_graph_key(sid))


@functools.lru_cache(maxsize=8)
def _graph_bytes(*key) -> bytes:
    return _graph(*key).model_dump_json().encode()


def graph_json(sid: int) -> bytes:
    """Serialized graph, cached per (run, edit_seq); a route can return it as a raw Response."""
    return _graph_bytes(*_graph_key(sid))


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
            q = q.where(or_(RequirementRow.text.ilike(pat, escape="\\"),
                            RequirementRow.req_id.ilike(pat, escape="\\")))
        total = s.scalar(select(func.count()).select_from(q.subquery()))
        col = _SORT.get(sort, RequirementRow.ordinal)
        col = col.desc() if dir == "desc" else col.asc()
        q = q.order_by(col, RequirementRow.ordinal).offset((page - 1) * page_size).limit(page_size)
        return [_req_out(*t) for t in s.execute(q)], total
