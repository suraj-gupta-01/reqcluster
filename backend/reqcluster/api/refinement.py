from fastapi import APIRouter

from reqcluster.api import PathId
from reqcluster.domain import Job
from reqcluster.services import refinement as svc

router = APIRouter()


@router.post("/sessions/{sid}/suggestions", status_code=202)
def suggest(sid: PathId, params: svc.SuggestParams | None = None) -> Job:
    return svc.start(sid, params or svc.SuggestParams())


@router.get("/sessions/{sid}/suggestions")
def list_suggestions(sid: PathId, status: str | None = None) -> list[svc.SuggestionOut]:
    return svc.list_suggestions(sid, status)


@router.post("/sessions/{sid}/suggestions/{suggestion_id}/accept")
def accept(sid: PathId, suggestion_id: PathId, body: svc.DecisionBody | None = None) -> svc.SuggestionOut:
    return svc.decide(sid, suggestion_id, True, body.actor if body else "")


@router.post("/sessions/{sid}/suggestions/{suggestion_id}/reject")
def reject(sid: PathId, suggestion_id: PathId, body: svc.DecisionBody | None = None) -> svc.SuggestionOut:
    return svc.decide(sid, suggestion_id, False, body.actor if body else "")


@router.get("/sessions/{sid}/audit")
def audit(sid: PathId) -> list[svc.AuditOut]:
    return svc.audit(sid)


@router.get("/sessions/{sid}/coherence")
def coherence(sid: PathId) -> list[svc.CoherenceOut]:
    return svc.coherence(sid)
