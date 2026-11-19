from typing import Literal

from fastapi import APIRouter, Query, Response, UploadFile

from reqcluster.api import PathClusterId, PathId
from reqcluster import ingest
from reqcluster.config import get_settings
from reqcluster.domain import ClusteringParams, ClusterRef, Job
from reqcluster.services import clustering, sessions

router = APIRouter()


@router.post("/sessions")
def upload(file: UploadFile) -> dict:
    s = get_settings()
    data = ingest.read_capped(file.file, s.max_upload_mb * 1024 * 1024)
    name = file.filename or ""
    res = ingest.parse(data, name, s.max_rows)
    out = sessions.create(res, name)
    return {"session": out, "ingest_report": {"stats": res.stats, "warnings": res.warnings}}


@router.get("/sessions")
def list_sessions() -> list[sessions.SessionOut]:
    return sessions.list_sessions()


@router.get("/sessions/{sid}")
def get_session(sid: PathId) -> sessions.SessionOut:
    return sessions.get(sid)


@router.delete("/sessions/{sid}", status_code=204)
def delete_session(sid: PathId) -> Response:
    sessions.delete(sid)
    return Response(status_code=204)


@router.post("/sessions/{sid}/cluster", status_code=202)
def cluster(sid: PathId, params: ClusteringParams | None = None) -> Job:
    return clustering.start(sid, params or ClusteringParams())


@router.get("/sessions/{sid}/clusters")
def clusters(sid: PathId) -> list[sessions.ClusterOut]:
    return sessions.clusters(sid)


@router.get("/sessions/{sid}/clusters/{cid}")
def cluster_detail(sid: PathId, cid: PathClusterId) -> sessions.ClusterDetail:
    return sessions.cluster_detail(sid, cid)


@router.get("/sessions/{sid}/graph")
def graph(sid: PathId) -> sessions.GraphOut:
    return sessions.graph(sid)


@router.get("/sessions/{sid}/metrics")
def metrics(sid: PathId) -> sessions.MetricsOut:
    return sessions.metrics(sid)


@router.get("/sessions/{sid}/requirements")
def requirements(
    sid: PathId,
    response: Response,
    cluster_id: ClusterRef | None = None,
    noise: bool | None = None,
    search: str | None = None,
    sort: Literal["ordinal", "req_id", "text", "cluster_id", "probability"] = "ordinal",
    dir: Literal["asc", "desc"] = "asc",
    page: int = Query(1, ge=1, le=2**31 - 1),
    page_size: int = Query(50, ge=1, le=1000),
) -> list[sessions.RequirementOut]:
    items, total = sessions.requirements(sid, cluster_id, noise, search, sort, dir, page, page_size)
    response.headers["X-Total-Count"] = str(total)
    return items
