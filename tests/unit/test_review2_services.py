"""Review-2 regressions: run embedding space (M7), ablation reuse (S8), batched narratives (S9),
cluster id never reused (S1), dependency tree staleness (S2)."""
from sqlalchemy import select

from reqcluster import embeddings
from reqcluster.db import session_scope
from reqcluster.domain import ClusteringParams
from reqcluster.llm import gateway as gw
from reqcluster.ml import pipeline
from reqcluster.models import AssignmentRow, ClusterRow, RunRow, SessionRow
from reqcluster.services import clustering, dependencies, enrichment, feedback, refinement, sessions

from .test_svc_clustering import make
from .test_svc_refinement import LOOSE

def _cur(sid):
    with session_scope() as s:
        return s.get(SessionRow, sid).current_run_id


ENR = "Domain-Aware Context"


def _record(monkeypatch):
    seen = []
    real = embeddings.embed
    monkeypatch.setattr(embeddings, "embed", lambda t, progress=None: seen.append(list(t)) or real(t))
    return seen


def _enriched_run(n=40, **p):
    sid = make(n)
    assert enrichment.start(sid, "mock").status == "succeeded"
    params = ClusteringParams(embedding_mode="hybrid", min_cluster_size=5, **p)
    assert clustering.start(sid, params).status == "succeeded"
    return sid


def test_follow_up_jobs_embed_the_run_space(monkeypatch):
    sid = _enriched_run()
    seen = _record(monkeypatch)
    assert refinement.start(sid, LOOSE).status == "succeeded"
    assert dependencies.start(sid, dependencies.DependencyRequest()).status == "succeeded"
    refinement.coherence(sid)
    assert feedback.recluster(sid).status == "succeeded"
    assert seen and all(ENR in t for batch in seen[:4] for t in batch)


def test_conflict_falls_back_to_base_with_warning(monkeypatch):
    sid = _enriched_run()
    # enrichment goes stale once requirement text changes: texts_for_mode raises Conflict
    monkeypatch.setattr(enrichment, "texts_for_mode", lambda s, m: (_ for _ in ()).throw(
        clustering.errors.Conflict("x")) if m != "base" else [f"b{i}" for i in range(40)])
    ws = []
    with session_scope() as s:
        run = s.get(RunRow, _cur(sid))
    clustering.run_embeddings(sid, run, ws)
    assert ws


def test_ablation_reuses_main_pipeline(monkeypatch):
    calls = []
    real = pipeline.run
    monkeypatch.setattr(pipeline, "run", lambda *a, **k: calls.append(1) or real(*a, **k))
    sid = make(40)
    enrichment.start(sid, "mock")
    clustering.start(sid, ClusteringParams(min_cluster_size=5, run_ablation=True))
    assert len(calls) == 2


def test_narratives_batched_in_one_map(monkeypatch):
    sid = make(60)
    clustering.start(sid, ClusteringParams(min_cluster_size=5))
    g = gw.get_gateway()
    n = []
    real = g.map
    monkeypatch.setattr(g, "map", lambda *a, **k: n.append(len(a[0])) or real(*a, **k))
    assert refinement.start(sid, LOOSE).status == "succeeded"
    assert len(n) == 1 and n[0] >= 1
    n.clear()
    assert dependencies.start(sid, dependencies.DependencyRequest()).status == "succeeded"
    assert len(n) == 1


def test_split_never_reuses_merged_away_id():
    sid = make(60)
    clustering.start(sid, ClusteringParams(min_cluster_size=5))
    with session_scope() as s:
        run = s.get(RunRow, _cur(sid))
        top = max(c.cluster_id for c in s.scalars(select(ClusterRow).where(ClusterRow.run_id == run.id)))
        ids = sorted(c.cluster_id for c in s.scalars(select(ClusterRow).where(ClusterRow.run_id == run.id)))
    src = top
    dst = ids[0]
    assert src != dst
    with session_scope() as s:
        rids = list(s.scalars(select(AssignmentRow.requirement_id).where(
            AssignmentRow.run_id == run.id, AssignmentRow.cluster_id == src)))
    for r in rids:
        feedback.submit(sid, r, dst)
    with session_scope() as s:
        run = s.get(RunRow, run.id)
        assert s.get(ClusterRow, (run.id, src)) is None
        assert clustering.next_cluster_id(s, run) == top + 1


def test_tree_stale_after_feedback_move():
    sid = make(30)
    clustering.start(sid, ClusteringParams(min_cluster_size=5))
    assert dependencies.start(sid, dependencies.DependencyRequest()).status == "succeeded"
    assert dependencies.get(sid).stale is False
    cl = [c.cluster_id for c in sessions.clusters(sid)]
    with session_scope() as s:
        a = s.scalars(select(AssignmentRow).where(
            AssignmentRow.run_id == _cur(sid),
            AssignmentRow.cluster_id == cl[0])).first()
        rid = a.requirement_id
    feedback.submit(sid, rid, cl[1])
    assert dependencies.get(sid).stale is True
