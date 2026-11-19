import pytest
from sqlalchemy import func, select

from reqcluster import errors
from reqcluster.db import init_db, session_scope
from reqcluster.domain import RequirementIn
from reqcluster.ingest import IngestResult
from reqcluster.models import EnrichmentRow
from reqcluster.services import enrichment, sessions


def make(n=4):
    init_db()
    reqs = [RequirementIn(req_id=f"R{i}", text=f"The brake system shall report status {i}",
                          ordinal=i) for i in range(n)]
    return sessions.create(IngestResult(reqs, {}), "t").id


def count():
    with session_scope() as s:
        return s.scalar(select(func.count()).select_from(EnrichmentRow))


def test_enrich_upserts_not_duplicates():
    sid = make()
    job = enrichment.start(sid, None)
    assert job.status == "succeeded"
    assert job.result["total"] == 4 and job.result["succeeded"] == 4 and job.result["failed"] == 0
    assert count() == 4
    enrichment.start(sid, None)
    assert count() == 4
    out = enrichment.results(sid)
    assert [o.req_id for o in out] == ["R0", "R1", "R2", "R3"] and out[0].status == "succeeded"


def test_texts_for_mode():
    sid = make()
    assert len(enrichment.texts_for_mode(sid, "base")) == 4
    with pytest.raises(errors.Conflict):
        enrichment.texts_for_mode(sid, "enriched")
    enrichment.start(sid, None)
    texts = enrichment.texts_for_mode(sid, "enriched")
    assert len(texts) == 4 and texts[0] != "The brake system shall report status 0"


def test_unknown_session():
    init_db()
    with pytest.raises(errors.NotFound):
        enrichment.start(999, None)
