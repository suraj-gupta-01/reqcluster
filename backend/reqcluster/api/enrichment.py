from fastapi import APIRouter

from reqcluster.api import PathId
from reqcluster.domain import Job
from reqcluster.services import enrichment

router = APIRouter()


@router.post("/sessions/{sid}/enrich", status_code=202)
def enrich(sid: PathId, body: enrichment.EnrichRequest | None = None) -> Job:
    return enrichment.start(sid, body.provider if body else None)


@router.get("/sessions/{sid}/enrichment")
def get_enrichment(sid: PathId) -> list[enrichment.EnrichmentOut]:
    return enrichment.results(sid)


@router.get("/sessions/{sid}/enrichment/summary")
def enrichment_summary(sid: PathId) -> enrichment.EnrichmentOverview:
    return enrichment.summary(sid)
