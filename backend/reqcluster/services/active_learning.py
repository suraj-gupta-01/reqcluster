"""Active learning: most uncertain requirements and per-run quality history."""
from datetime import datetime

from pydantic import BaseModel
from sqlalchemy import select

from reqcluster.db import session_scope
from reqcluster.errors import NotFound
from reqcluster.ml import active
from reqcluster.models import RequirementRow, RunRow, SessionRow
from reqcluster.services.feedback import _ordered, load_npz
from reqcluster.services.sessions import _current_run


class UncertainItem(BaseModel):
    requirement_id: int
    req_id: str
    text: str
    cluster_id: int
    is_noise: bool
    membership_prob: float
    uncertainty: float


class QualityPoint(BaseModel):
    revision: int
    kind: str
    silhouette: float | None
    n_clusters: int
    noise_count: int
    created_at: datetime


def uncertainty(sid: int, top_k: int = 20) -> list[UncertainItem]:
    with session_scope() as s:
        run = _current_run(s, sid)
        pks, _, labels = _ordered(s, run.id, sid)
        rid = run.id
        info = {r.id: (r.req_id, r.text) for r in s.scalars(
            select(RequirementRow).where(RequirementRow.session_id == sid))}
    npz = load_npz(rid)
    top = active.uncertainty(npz["probabilities"], npz["x10"], labels, top_k)
    return [UncertainItem(requirement_id=pks[t["index"]], req_id=info[pks[t["index"]]][0],
                          text=info[pks[t["index"]]][1], cluster_id=t["cluster_id"],
                          is_noise=t["is_noise"], membership_prob=t["membership_prob"],
                          uncertainty=t["uncertainty"]) for t in top]


def quality(sid: int) -> list[QualityPoint]:
    with session_scope() as s:
        if s.get(SessionRow, sid) is None:
            raise NotFound(f"session {sid} not found")
        runs = s.scalars(select(RunRow).where(RunRow.session_id == sid)
                         .order_by(RunRow.revision)).all()
        return [QualityPoint(revision=r.revision, kind=r.kind,
                             silhouette=(r.metrics or {}).get("silhouette"),
                             n_clusters=r.n_clusters, noise_count=r.noise_count,
                             created_at=r.created_at) for r in runs]
