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
    assert out[0].text == "The brake system shall report status 0"
    assert enrichment.summary(sid).provider == out[0].provider


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


def test_real_provider_fallback_is_counted_and_not_used_for_enriched_clustering(monkeypatch):
    from reqcluster.config import get_settings
    from reqcluster.llm import gateway as gw

    monkeypatch.setenv("REQCLUSTER_LLM_BASE_URL", "")
    monkeypatch.setenv("REQCLUSTER_LLM_MAX_RETRIES", "0")
    get_settings.cache_clear()
    gw.reset_gateway()
    sid = make()
    job = enrichment.start(sid, "openai")
    assert job.result["fallback"] == 4 and job.result["succeeded"] == 4
    with pytest.raises(errors.Conflict):  # template text must not silently feed hybrid clustering
        enrichment.texts_for_mode(sid, "hybrid")


def test_cancel_after_first_chunk_keeps_saved_rows(monkeypatch):
    from reqcluster.llm import enrich as le

    real = le.expand_all

    def small(reqs, gw, provider=None, on_chunk=None, chunk=32):
        return real(reqs, gw, provider, on_chunk, chunk=2)

    monkeypatch.setattr(enrichment, "expand_all", small)
    sid = make(5)
    calls = []
    orig = enrichment.JobContext.check_cancel

    def cancel_after_first(self):
        calls.append(1)
        if len(calls) == 1:
            from reqcluster.db import session_scope as ss
            from reqcluster.models import JobRow
            with ss() as s:
                s.get(JobRow, self.job_id).cancel_requested = True
        orig(self)

    monkeypatch.setattr(enrichment.JobContext, "check_cancel", cancel_after_first)
    job = enrichment.start(sid, None)
    assert job.status == "cancelled" and count() == 2


def test_summary_counts_failed_and_fallback(monkeypatch):
    from reqcluster.config import get_settings
    from reqcluster.llm import gateway as gw

    monkeypatch.setenv("REQCLUSTER_LLM_BASE_URL", "")
    monkeypatch.setenv("REQCLUSTER_LLM_MAX_RETRIES", "0")
    get_settings.cache_clear()
    gw.reset_gateway()
    sid = make()
    enrichment.start(sid, "openai")
    o = enrichment.summary(sid)
    assert (o.total, o.succeeded, o.failed, o.fallback) == (4, 4, 0, 4)


def test_summary_reports_an_all_failed_enrichment(monkeypatch):
    # Overview used to show nothing when every row failed (summary raised 409)
    from types import SimpleNamespace

    from reqcluster.llm.gateway import LLMResult
    from reqcluster.services import enrichment as svc

    class BadLLM:  # every answer fails validation -> every expansion is "failed"
        def map(self, prompts, fallback_for, provider=None):
            return [LLMResult(SimpleNamespace(model_dump=lambda: {}), "llm", "openai") for _ in prompts]

    monkeypatch.setattr(svc, "get_gateway", lambda: BadLLM())
    sid = make()
    enrichment.start(sid, "openai")
    out = enrichment.summary(sid)
    assert out.total == 4 and out.succeeded == 0 and out.failed == 4
