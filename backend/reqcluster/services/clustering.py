"""Clustering job: embed, reduce, cluster, then persist one revision in a single transaction."""
import contextlib
import json
import os
import tempfile
from dataclasses import asdict

import numpy as np
from pydantic import BaseModel
from sqlalchemy import delete, func, insert, select

from reqcluster import embeddings
from reqcluster.config import get_settings
from reqcluster.db import session_scope
from reqcluster.domain import ClusteringParams, Job
from reqcluster import errors
from reqcluster.errors import NotFound
from reqcluster.jobs import JobContext, get_runner
from reqcluster.ml import analysis, metrics, pipeline
from reqcluster.models import (
    AssignmentRow,
    ClusterRow,
    ConstraintRow,
    RequirementRow,
    RunRow,
    SessionRow,
)
from reqcluster.services import enrichment
from reqcluster.services.sessions import unlink_runs

KEEP_RUNS = 5


class ClusteringSummary(BaseModel):
    run_id: int
    revision: int
    n_clusters: int
    noise_count: int
    warnings: list[str]
    durations: dict
    metrics: dict


def start(sid: int, params: ClusteringParams) -> Job:
    if params.embedding_mode != "base":
        enrichment.texts_for_mode(sid, params.embedding_mode)  # 404 / fail fast (409) if incomplete
    else:
        with session_scope() as s:
            if s.get(SessionRow, sid) is None:
                raise NotFound(f"session {sid} not found")
    return get_runner().submit("cluster", sid, lambda ctx: _run(ctx, sid, params),
                               params.model_dump())


def run_embeddings(sid: int, run: RunRow, warnings: list[str] | None = None) -> np.ndarray:
    """Embeddings of the run's requirements (ordinal order) in the space it was clustered in."""
    mode = (run.params or {}).get("embedding_mode", "base")
    try:
        texts = enrichment.texts_for_mode(sid, mode)
    except errors.Conflict:
        texts = enrichment.texts_for_mode(sid, "base")
        if warnings is not None:
            warnings.append(f"Enrichment is incomplete: used base text embeddings, not {mode}.")
    return embeddings.embed(texts)


def next_cluster_id(s, run: RunRow) -> int:
    """Fresh cluster id, never one the run has used before (merged-away ids stay retired)."""
    live = s.scalar(select(func.max(ClusterRow.cluster_id)).where(ClusterRow.run_id == run.id))
    run.max_cluster_id = max(run.max_cluster_id, live if live is not None else -1) + 1
    return run.max_cluster_id


def _scaled(ctx: JobContext, lo: float, hi: float, step: str):
    return lambda frac, name=step: ctx.progress(lo + (hi - lo) * frac, name)


def _run(ctx: JobContext, sid: int, params: ClusteringParams) -> ClusteringSummary:
    settings = get_settings()
    with session_scope() as s:
        rows = s.execute(select(RequirementRow.id, RequirementRow.text)
                         .where(RequirementRow.session_id == sid)
                         .order_by(RequirementRow.ordinal)).all()
    req_ids = [r[0] for r in rows]
    texts = [r[1] for r in rows]

    # enriched/hybrid change only what is embedded; keywords and the graph keep the raw text
    embed_texts = (texts if params.embedding_mode == "base"
                   else enrichment.texts_for_mode(sid, params.embedding_mode))
    ctx.progress(0.05, "embedding")
    emb = embeddings.embed(
        embed_texts, progress=lambda done, total: ctx.progress(0.05 + 0.55 * done / max(total, 1),
                                                         "embedding"))
    ctx.check_cancel()
    seed = params.seed if params.seed is not None else settings.seed
    res = pipeline.run(emb, texts, params, seed=seed, threads=settings.threads,
                       progress=lambda frac, step: ctx.progress(0.6 + 0.35 * frac, step))
    ctx.check_cancel()
    m = metrics.compute(res.labels, res.x10, seed)

    edges = [[req_ids[i], req_ids[j], float(w)] for i, j, w in res.edges]
    reports = _reports(ctx, sid, params, texts, emb, res, seed, settings.threads)
    rid, revision = persist_run(sid, "cluster", params.model_dump(), req_ids, res.labels,
                                res.probabilities, res.x10, res.xy, res.clusters, edges, m,
                                res.warnings, res.durations, reports=reports)
    return ClusteringSummary(run_id=rid, revision=revision, n_clusters=len(res.clusters),
                             noise_count=int((res.labels == -1).sum()), warnings=res.warnings,
                             durations=res.durations, metrics=m)


def _reports(ctx, sid, params, texts, emb, res, seed, threads) -> dict:
    if not (params.enable_embedding_comparison or params.run_ablation):
        return {}
    try:
        enriched = enrichment.texts_for_mode(sid, "enriched")
    except errors.Conflict:
        res.warnings.append("Embedding comparison/ablation skipped: enrichment is incomplete.")
        return {}
    out = {}
    if params.enable_embedding_comparison:
        ctx.progress(0.96, "embedding comparison")
        base = emb if params.embedding_mode == "base" else embeddings.embed(texts)
        cmp = analysis.compare_embeddings(base, embeddings.embed(enriched),
                                          labels_base=res.labels, seed=seed)
        out["embedding_comparison"] = asdict(cmp)
    if params.run_ablation:
        ctx.progress(0.98, "ablation")
        p = params.model_copy(update={"embedding_mode": "enriched"}) \
            if params.embedding_mode == "base" else params
        out["ablation_report"] = asdict(analysis.run_ablation(
            texts, enriched, p, embeddings.embed, seed, threads,
            main=(params.embedding_mode, emb, res)))
    return json.loads(json.dumps(out, default=float))  # numpy scalars / int keys -> JSON-safe


def persist_run(sid, kind, params, req_ids, labels, probabilities, x10, xy, clusters, edges,
                metrics, warnings, durations, constraints=(), reports=None) -> tuple[int, int]:
    """Write one new revision (rows + npz + carried constraints), make it current, prune old
    runs. One transaction, so a crash can never leave a constrained run without its constraints."""
    n_noise = int((labels == -1).sum())
    with session_scope() as s:
        revision = (s.scalar(select(func.max(RunRow.revision)).where(RunRow.session_id == sid))
                    or 0) + 1
        run = RunRow(session_id=sid, revision=revision, kind=kind, params=params,
                     metrics=metrics, warnings=warnings, durations=durations,
                     reports=reports or {}, n_clusters=len(clusters), noise_count=n_noise,
                     max_cluster_id=int(labels.max()) if len(labels) else -1)
        s.add(run)
        s.flush()
        rid = run.id
        s.execute(insert(AssignmentRow), [
            {"run_id": rid, "requirement_id": req_ids[i], "cluster_id": int(labels[i]),
             "probability": float(probabilities[i]), "x": float(xy[i, 0]), "y": float(xy[i, 1])}
            for i in range(len(req_ids))
        ])
        if clusters:
            s.execute(insert(ClusterRow), [
                {"run_id": rid, "cluster_id": c.cluster_id, "size": c.size, "label": c.label,
                 "keywords": c.keywords} for c in clusters])
        if constraints:
            s.execute(insert(ConstraintRow), [
                {"run_id": rid, "a_id": a, "b_id": b, "kind": k, "feedback_id": f}
                for a, b, k, f in constraints])
        _write_npz(rid, x10=x10, xy=xy, probabilities=probabilities,
                   edges=np.asarray(edges, dtype=np.float64).reshape(-1, 3))
        s.get(SessionRow, sid).current_run_id = rid
        old = list(s.scalars(select(RunRow.id).where(RunRow.session_id == sid)
                             .order_by(RunRow.revision.desc()).offset(KEEP_RUNS)))
        if old:
            s.execute(delete(RunRow).where(RunRow.id.in_(old)))
    unlink_runs(old)
    return rid, revision


def load_run_arrays(run_id: int) -> dict[str, np.ndarray]:
    """Per-run arrays: x10, xy, probabilities, edges ([src_pk, dst_pk, weight] rows).

    Read fully and closed at once (an open npz blocks deletion on Windows)."""
    with np.load(get_settings().runs_dir / f"{run_id}.npz") as z:
        return {k: z[k] for k in z.files}


def _write_npz(run_id: int, **arrays) -> None:
    d = get_settings().runs_dir
    d.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            np.savez_compressed(f, **arrays)
        os.replace(tmp, d / f"{run_id}.npz")
    except BaseException:
        with contextlib.suppress(OSError):  # Windows may refuse while the file is open
            os.unlink(tmp)
        raise
