"""Real-LLM smoke: enrichment, suggestion rationales and an edge explanation through the
configured provider (e.g. Ollama). Reports successes vs fallbacks, latency and samples.

    REQCLUSTER_LLM_PROVIDER=local uv run python scripts/llm_smoke.py --n 8
"""
import argparse
import os
import tempfile
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--csv", default="data/aerospace_requirements.csv")
    args = ap.parse_args()
    os.environ.setdefault("REQCLUSTER_HOME", tempfile.mkdtemp(prefix="llm_smoke_"))
    os.environ.setdefault("REQCLUSTER_JOB_WORKERS", "0")

    from reqcluster.config import get_settings
    from reqcluster.db import init_db
    from reqcluster.domain import ClusteringParams
    from reqcluster.ingest import parse
    from reqcluster.services import clustering, dependencies, enrichment, refinement, sessions

    s = get_settings()
    print(f"provider={s.llm_provider} model={s.local_llm_model if s.llm_provider == 'local' else s.llm_model}")
    init_db()
    res = parse(Path(args.csv).read_bytes(), Path(args.csv).name)
    res.requirements = res.requirements[: max(args.n, 20)]
    sid = sessions.create(res, "llm-smoke").id

    t = time.perf_counter()
    job = enrichment.start(sid, None)
    r = job.result or {}
    print(f"enrich: {job.status} {time.perf_counter() - t:.1f}s total={r.get('total')} "
          f"succeeded={r.get('succeeded')} fallback={r.get('fallback')} failed={r.get('failed')}")
    for row in enrichment.results(sid)[:2]:
        print(f"  [{row.source}] {row.expanded_text[:160]!r} terms={row.domain_terms[:5]}")

    clustering.start(sid, ClusteringParams())
    t = time.perf_counter()
    job = refinement.start(sid, refinement.SuggestParams(sim_threshold=0.0, min_silhouette_delta=None))
    print(f"suggest: {job.status} {time.perf_counter() - t:.1f}s")
    for sug in refinement.list_suggestions(sid)[:2]:
        print(f"  [{sug.provider}] {sug.kind}: {sug.rationale[:200]!r}")

    t = time.perf_counter()
    job = dependencies.start(sid, dependencies.DependencyRequest())
    tree = dependencies.get(sid)
    print(f"dependencies: {job.status} {time.perf_counter() - t:.1f}s edges={len(tree.edges)}")
    if tree.edges:
        e = tree.edges[0]
        t = time.perf_counter()
        ex = dependencies.explain(sid, dependencies.ExplainRequest(source=e["source"], target=e["target"]))
        print(f"explain: {time.perf_counter() - t:.1f}s provider={ex.provider} -> {ex.explanation[:240]!r}")


if __name__ == "__main__":
    main()
