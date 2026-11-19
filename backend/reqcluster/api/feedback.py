from typing import Literal

from fastapi import APIRouter, Query, Response
from pydantic import BaseModel

from reqcluster.domain import Job
from reqcluster.services import active_learning, feedback

router = APIRouter()


class FeedbackIn(BaseModel):
    requirement_id: int
    new_cluster_id: int | None = None
    comments: str = ""
    applied_by: str = ""


@router.post("/sessions/{sid}/feedback")
def submit(sid: int, body: FeedbackIn) -> feedback.SubmitOut:
    return feedback.submit(sid, body.requirement_id, body.new_cluster_id, body.comments,
                           body.applied_by)


@router.get("/sessions/{sid}/feedback")
def queue(sid: int, status: Literal["pending", "approved", "rejected"] | None = None
          ) -> list[feedback.FeedbackOut]:
    return feedback.queue(sid, status)


@router.get("/sessions/{sid}/feedback/export")
def export(sid: int, format: Literal["csv", "json"] = "csv") -> Response:
    body, media = feedback.export(sid, format)
    return Response(body, media_type=media)


@router.post("/sessions/{sid}/feedback/{fid}/approve")
def approve(sid: int, fid: int) -> feedback.FeedbackOut:
    return feedback.review(sid, fid, True)


@router.post("/sessions/{sid}/feedback/{fid}/reject")
def reject(sid: int, fid: int) -> feedback.FeedbackOut:
    return feedback.review(sid, fid, False)


@router.get("/sessions/{sid}/constraints")
def constraints(sid: int) -> feedback.ConstraintsOut:
    return feedback.constraints(sid)


@router.post("/sessions/{sid}/recluster", status_code=202)
def recluster(sid: int) -> Job:
    return feedback.recluster(sid)


@router.get("/sessions/{sid}/uncertainty")
def uncertainty(sid: int, top_k: int = Query(20, ge=1, le=1000)
                ) -> list[active_learning.UncertainItem]:
    return active_learning.uncertainty(sid, top_k)


@router.get("/sessions/{sid}/quality")
def quality(sid: int) -> list[active_learning.QualityPoint]:
    return active_learning.quality(sid)
