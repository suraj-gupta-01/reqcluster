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
from reqcluster.llm.narrate import cluster_narratives, explain_edge
from reqcluster.ml import refine
from reqcluster.ml.dependency.tree import build
from reqcluster.models import (
    AssignmentRow, ClusterRow, DependencyExplanationRow, DependencyTreeRow, RequirementRow, SessionRow,
)
from reqcluster.services.clustering import run_embeddings
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
    rationale: dict = {}
    stale: bool = False  # clusters were edited after the tree was built
    warnings: list[str] = []


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
        run = _current_run(s, sid)
        run_id, edit_seq = run.id, run.edit_seq
        rows = s.execute(select(RequirementRow.id, RequirementRow.req_id, RequirementRow.text,
                                AssignmentRow.cluster_id)
                         .join(AssignmentRow, (AssignmentRow.requirement_id == RequirementRow.id)
                               & (AssignmentRow.run_id == run_id))
                         .where(RequirementRow.session_id == sid)
                         .order_by(RequirementRow.ordinal)).all()
        clusters = s.scalars(select(ClusterRow).where(ClusterRow.run_id == run_id)
                             .order_by(ClusterRow.cluster_id)).all()
        clusters = [(c.cluster_id, c.label, list(c.keywords or []), c.size) for c in clusters]
    pks = [r[0] for r in rows]
    texts = [r[2] for r in rows]
    ctx.progress(0.1, "embedding")
    warnings: list[str] = []
    emb = run_embeddings(sid, run, warnings)
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
    labels = np.array([r[3] for r in rows])
    reps = refine.representatives(emb, texts, labels, [r[1] for r in rows], top_n=3)
    items = [{"kind": "summary", "cluster_label": label, "keywords": kws,
              "representative_texts": [r.text for r in reps.get(cid, [])]}
             for cid, label, kws, _ in clusters]
    summaries = cluster_narratives(get_gateway(), items, req.provider)
    grouping = [{"cluster_id": cid, "label": label, "keywords": kws, "size": size,
                 "representative_req_ids": [r.req_id for r in reps.get(cid, [])],
                 "rationale": str(sm.output)}
                for (cid, label, kws, size), sm in zip(clusters, summaries, strict=True)]
    rid_of = {r[0]: r[1] for r in rows}
    rationale = {"grouping": grouping, "stats": tree["stats"], "dependencies": [
        {"source": e["source"], "target": e["target"], "source_req_id": rid_of[e["source"]],
         "target_req_id": rid_of[e["target"]], "relation": e["relation"], "weight": e["weight"],
         "justification": e["rationale"]} for e in tree["edges"]]}
    with session_scope() as s:
        if s.get(SessionRow, sid).current_run_id != run_id:
            raise Conflict("session was re-clustered while the tree was building")
        s.execute(delete(DependencyTreeRow).where(DependencyTreeRow.run_id == run_id))
        s.add(DependencyTreeRow(run_id=run_id, params={**req.model_dump(exclude_none=True), "edit_seq": edit_seq},
                                nodes=tree["nodes"], edges=tree["edges"], stats=tree["stats"],
                                rationale=rationale))
    return TreeOut(run_id=run_id, rationale=rationale, warnings=warnings, **tree)


def get(sid: int) -> TreeOut:
    with session_scope() as s:
        run = _current_run(s, sid)
        run_id = run.id
        row = s.get(DependencyTreeRow, run_id)
        if row is None:
            raise Conflict("no dependency tree for the current run; generate one first")
        return TreeOut(run_id=run_id, nodes=row.nodes, edges=row.edges, stats=row.stats,
                       rationale=row.rationale or {},
                       stale=row.params.get("edit_seq") != run.edit_seq)


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
