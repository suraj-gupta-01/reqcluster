"""Dependency tree (run-scoped) and per-edge LLM explanations."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import delete, select

from reqcluster import embeddings
from reqcluster.config import get_settings
from reqcluster.db import session_scope
from reqcluster.domain import Job, ProviderName
from reqcluster.errors import Conflict, NotFound
from reqcluster.jobs import JobContext, get_runner
from reqcluster.llm.gateway import get_gateway
from reqcluster.llm.narrate import explain_edge
from reqcluster.ml.dependency.tree import build
from reqcluster.models import (
    AssignmentRow, DependencyExplanationRow, DependencyTreeRow, RequirementRow, SessionRow,
)
from reqcluster.services.sessions import _current_run


class DependencyRequest(BaseModel):
    model_config = {"extra": "forbid"}
    provider: ProviderName | None = None
    sim_threshold: float | None = Field(None, ge=0, le=1)
    top_k: int | None = Field(None, ge=1)


class ExplainRequest(BaseModel):
    model_config = {"extra": "forbid"}
    source: int
    target: int
    provider: ProviderName | None = None
    force: bool = False


class TreeOut(BaseModel):
    run_id: int
    nodes: list
    edges: list
    stats: dict


class ExplanationOut(BaseModel):
    source: int
    target: int
    explanation: str
    provider: str
    cached: bool


def start(sid: int, req: DependencyRequest) -> Job:
    with session_scope() as s:
        _current_run(s, sid)
    return get_runner().submit("dependencies", sid, lambda ctx: _run(ctx, sid, req),
                               req.model_dump(exclude_none=True))


def _run(ctx: JobContext, sid: int, req: DependencyRequest) -> TreeOut:
    settings = get_settings()
    with session_scope() as s:
        run_id = _current_run(s, sid).id
        rows = s.execute(select(RequirementRow.id, RequirementRow.req_id, RequirementRow.text,
                                AssignmentRow.cluster_id)
                         .join(AssignmentRow, (AssignmentRow.requirement_id == RequirementRow.id)
                               & (AssignmentRow.run_id == run_id))
                         .where(RequirementRow.session_id == sid)
                         .order_by(RequirementRow.ordinal)).all()
    pks = [r[0] for r in rows]
    texts = [r[2] for r in rows]
    ctx.progress(0.1, "embedding")
    emb = embeddings.embed(texts)
    ctx.check_cancel()
    kw = {k: v for k, v in (("sim_threshold", req.sim_threshold), ("top_k", req.top_k))
          if v is not None}
    ctx.progress(0.5, "building")
    tree = build(emb, texts, [r[1] for r in rows], np.array([r[3] for r in rows]),
                 embed_fn=lambda t: embeddings.embed(t), spacy_model=settings.spacy_model,
                 disable_spacy=settings.disable_spacy, **kw).to_dict()
    for n in tree["nodes"]:
        n["id"] = pks[n["id"]]
    for e in tree["edges"]:
        e["source"], e["target"] = pks[e["source"]], pks[e["target"]]
    with session_scope() as s:
        if s.get(SessionRow, sid).current_run_id != run_id:
            raise Conflict("session was re-clustered while the tree was building")
        s.execute(delete(DependencyTreeRow).where(DependencyTreeRow.run_id == run_id))
        s.add(DependencyTreeRow(run_id=run_id, params=req.model_dump(exclude_none=True),
                                nodes=tree["nodes"], edges=tree["edges"], stats=tree["stats"]))
    return TreeOut(run_id=run_id, **tree)


def get(sid: int) -> TreeOut:
    with session_scope() as s:
        run_id = _current_run(s, sid).id
        row = s.get(DependencyTreeRow, run_id)
        if row is None:
            raise Conflict("no dependency tree for the current run; generate one first")
        return TreeOut(run_id=run_id, nodes=row.nodes, edges=row.edges, stats=row.stats)


def explain(sid: int, req: ExplainRequest) -> ExplanationOut:
    with session_scope() as s:
        run_id = _current_run(s, sid).id
        tree = s.get(DependencyTreeRow, run_id)
        edge = next((e for e in (tree.edges if tree else [])
                     if e["source"] == req.source and e["target"] == req.target), None)
        if edge is None:
            raise NotFound(f"no dependency edge {req.source} -> {req.target}")
        if not req.force:
            hit = s.get(DependencyExplanationRow, (run_id, req.source, req.target))
            if hit is not None:
                return ExplanationOut(source=req.source, target=req.target,
                                      explanation=hit.explanation["text"],
                                      provider=hit.provider, cached=True)
        reqs = {r.id: r for r in s.scalars(select(RequirementRow).where(
            RequirementRow.id.in_((req.source, req.target))))}
    src, tgt = reqs[req.source], reqs[req.target]
    res = explain_edge(get_gateway(), src.text, tgt.text, edge["relation"], edge.get("evidence"),
                       source_id=src.req_id, target_id=tgt.req_id,
                       confidence_tier=edge.get("confidence_tier", ""),
                       rationale=edge.get("rationale", ""), provider=req.provider)
    text = str(res.output)
    with session_scope() as s:
        if s.get(SessionRow, sid).current_run_id == run_id:
            s.execute(delete(DependencyExplanationRow).where(
                DependencyExplanationRow.run_id == run_id,
                DependencyExplanationRow.source == req.source,
                DependencyExplanationRow.target == req.target))
            s.add(DependencyExplanationRow(run_id=run_id, source=req.source, target=req.target,
                                           provider=res.provider, explanation={"text": text}))
    return ExplanationOut(source=req.source, target=req.target, explanation=text,
                          provider=res.provider, cached=False)
