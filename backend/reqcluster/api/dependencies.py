from fastapi import APIRouter

from reqcluster.domain import Job
from reqcluster.services import dependencies as svc

router = APIRouter()


@router.post("/sessions/{sid}/dependencies", status_code=202)
def generate(sid: int, body: svc.DependencyRequest | None = None) -> Job:
    return svc.start(sid, body or svc.DependencyRequest())


@router.get("/sessions/{sid}/dependencies")
def get_tree(sid: int) -> svc.TreeOut:
    return svc.get(sid)


@router.post("/sessions/{sid}/dependencies/explain")
def explain(sid: int, body: svc.ExplainRequest) -> svc.ExplanationOut:
    return svc.explain(sid, body)
