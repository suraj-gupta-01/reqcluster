"""LLM enrichment: session-scoped expansions that survive re-clustering."""
from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from pydantic import BaseModel
from sqlalchemy import insert, select

from reqcluster.config import get_settings
from reqcluster.db import session_scope
from reqcluster.domain import Job, ProviderName
from reqcluster.errors import Conflict
from reqcluster.jobs import JobContext, get_runner
from reqcluster.llm.enrich import (
    PROMPT_VERSION, Expansion, build_quality_report, evaluate_expansion_quality, expand_all,
    extract_domain_vocabulary_details,
)
from reqcluster.llm.gateway import get_gateway
from reqcluster.models import EnrichmentRow, RequirementRow
from reqcluster.services.sessions import _row
from reqcluster.texts import build_texts


class EnrichRequest(BaseModel):
    model_config = {"extra": "forbid"}
    provider: ProviderName | None = None


class EnrichmentSummary(BaseModel):
    total: int
    succeeded: int
    failed: int
    fallback: int = 0  # succeeded via the template fallback because the real provider failed
    provider: str


class EnrichmentOut(BaseModel):
    requirement_id: int
    req_id: str
    text: str
    expanded_text: str
    domain_terms: list
    functional_intent: str
    mentioned_components: list
    assumptions: list
    confidence: float | None
    warnings: list
    status: str
    source: str
    provider: str
    quality: dict = {}


class EnrichmentOverview(BaseModel):
    total: int
    succeeded: int
    failed: int
    fallback: int = 0
    quality_report: dict
    domain_vocabulary: list[dict]
    provider: str


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def start(sid: int, provider: ProviderName | None) -> Job:
    with session_scope() as s:
        _row(s, sid)
    return get_runner().submit("enrich", sid, lambda ctx: _run(ctx, sid, provider),
                               {"provider": provider})


def _run(ctx: JobContext, sid: int, provider: ProviderName | None) -> EnrichmentSummary:
    provider = provider or get_settings().llm_provider
    with session_scope() as s:
        reqs = [(r.id, r.text) for r in s.scalars(
            select(RequirementRow).where(RequirementRow.session_id == sid)
            .order_by(RequirementRow.ordinal))]
    ctx.progress(0.02, "enriching")
    text_of = dict(reqs)
    ok = fb = n = 0

    def save(done: int, total: int, new: list[Expansion]) -> None:
        nonlocal ok, fb, n
        _persist(sid, provider, text_of, new)
        n += len(new)
        ok += sum(e.status == "succeeded" for e in new)
        fb += sum(e.status == "succeeded" and _is_fallback(e.source, provider) for e in new)
        ctx.progress(0.02 + 0.96 * done / max(1, total), "enriching", f"{done}/{total}")
        ctx.check_cancel()

    expand_all(reqs, get_gateway(), provider, on_chunk=save)
    return EnrichmentSummary(total=n, succeeded=ok, failed=n - ok, fallback=fb, provider=provider)


def _persist(sid: int, provider: str, text_of: dict[int, str], results: list[Expansion]) -> None:
    rows = [{
        "session_id": sid, "requirement_id": e.requirement_id, "provider": provider,
        "prompt_version": PROMPT_VERSION, "text_hash": _hash(text_of[e.requirement_id]),
        "status": e.status, "source": e.source, "expanded_text": e.expanded_text,
        "domain_terms": e.domain_terms, "functional_intent": e.functional_intent,
        "mentioned_components": e.mentioned_components, "assumptions": e.assumptions,
        "confidence": e.confidence, "warnings": e.warnings,
        "quality": evaluate_expansion_quality(text_of[e.requirement_id], e.expanded_text,
                                              e.domain_terms, e.confidence)
        if e.status == "succeeded" else {}, "error": e.error, "created_at": datetime.now(UTC),
    } for e in results]
    with session_scope() as s:
        have = {e.requirement_id: e for e in s.scalars(select(EnrichmentRow).where(
            EnrichmentRow.session_id == sid, EnrichmentRow.provider == provider,
            EnrichmentRow.prompt_version == PROMPT_VERSION))}
        fresh = []
        for row in rows:
            cur = have.get(row["requirement_id"])
            if cur is None:
                fresh.append(row)
            else:
                for k, v in row.items():
                    setattr(cur, k, v)
        if fresh:
            s.execute(insert(EnrichmentRow), fresh)


def _is_fallback(source: str, provider: str | None) -> bool:
    """Template text produced because a real provider failed (mock output is the intended text)."""
    return source == "fallback" and (provider or "mock") != "mock"


def results(sid: int) -> list[EnrichmentOut]:
    with session_scope() as s:
        _row(s, sid)
        reqs = {r.id: r for r in s.scalars(select(RequirementRow)
                                           .where(RequirementRow.session_id == sid))}
        latest: dict[int, EnrichmentRow] = {}
        for e in s.scalars(select(EnrichmentRow).where(EnrichmentRow.session_id == sid)
                           .order_by(EnrichmentRow.created_at, EnrichmentRow.id)):
            latest[e.requirement_id] = e
        return [EnrichmentOut(
            requirement_id=rid, req_id=reqs[rid].req_id, text=reqs[rid].text, expanded_text=e.expanded_text,
            domain_terms=e.domain_terms, functional_intent=e.functional_intent,
            mentioned_components=e.mentioned_components, assumptions=e.assumptions,
            confidence=e.confidence, warnings=e.warnings, status=e.status, source=e.source,
            provider=e.provider, quality=e.quality or {})
            for rid, e in sorted(latest.items(), key=lambda kv: reqs[kv[0]].ordinal)]


def texts_for_mode(sid: int, mode: str) -> list[str]:
    with session_scope() as s:
        _row(s, sid)
        reqs = list(s.scalars(select(RequirementRow).where(RequirementRow.session_id == sid)
                              .order_by(RequirementRow.ordinal)))
        enr: dict[int, str] = {}
        by_id = {r.id: r for r in reqs}
        if mode != "base":
            for e in s.scalars(select(EnrichmentRow).where(
                    EnrichmentRow.session_id == sid, EnrichmentRow.status == "succeeded")
                    .order_by(EnrichmentRow.created_at, EnrichmentRow.id)):
                r = by_id.get(e.requirement_id)
                if r is not None and e.text_hash == _hash(r.text) and not _is_fallback(e.source,
                                                                                       e.provider):
                    enr[r.id] = e.expanded_text
        texts = [r.text for r in reqs]
        if mode == "base":
            return build_texts(texts, mode)[0]
        if len(enr) < len(reqs):
            raise Conflict(f"{len(reqs) - len(enr)} requirement(s) lack current enrichment")
        return build_texts(texts, mode, [enr[r.id] for r in reqs])[0]


def summary(sid: int) -> EnrichmentOverview:
    with session_scope() as s:
        _row(s, sid)
        reqs = {r.id: r for r in s.scalars(select(RequirementRow)
                                           .where(RequirementRow.session_id == sid))}
        latest: dict[int, EnrichmentRow] = {}
        for e in s.scalars(select(EnrichmentRow).where(EnrichmentRow.session_id == sid)
                           .order_by(EnrichmentRow.created_at, EnrichmentRow.id)):
            latest[e.requirement_id] = e
        if not latest:
            raise Conflict("no enrichment yet; run enrichment first")
        every = sorted(latest.items(), key=lambda kv: reqs[kv[0]].ordinal)
        failed = sum(e.status != "succeeded" for _, e in every)
        fallback = sum(e.status == "succeeded" and _is_fallback(e.source, e.provider)
                       for _, e in every)
        items = [(r, e) for r, e in every if e.status == "succeeded"]
        provider = every[-1][1].provider  # an all-failed run is still reported, with its counts
        pairs = [(rid, reqs[rid].text) for rid, _ in items]
        exps = [Expansion(rid, e.expanded_text, e.domain_terms, e.functional_intent,
                          e.mentioned_components, e.assumptions, e.confidence or 0.0,
                          e.warnings, "succeeded", e.source) for rid, e in items]
    return EnrichmentOverview(
        total=len(every), succeeded=len(items), failed=failed, fallback=fallback,
        quality_report=build_quality_report(pairs, exps),
        domain_vocabulary=extract_domain_vocabulary_details([x.expanded_text for x in exps]),
        provider=provider)
