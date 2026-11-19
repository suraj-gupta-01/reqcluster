from fastapi import APIRouter

from reqcluster.domain import Job
from reqcluster.jobs import get_runner

router = APIRouter()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    return get_runner().get(job_id)


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> Job:
    return get_runner().cancel(job_id)
