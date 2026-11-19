"""Profile the heavy read paths (graph, CSV export) on a synthetic N-row clustered session.

Builds the session through the real services (sessions.create + clustering.persist_run with
random arrays, no ML), then times each read cold and prints cProfile's top self-time entries.
    uv run python scripts/bench_reads.py --n 35000
"""
import argparse
import cProfile
import os
import pstats
import tempfile
import time

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=35000)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()
    os.environ.setdefault("REQCLUSTER_HOME", tempfile.mkdtemp(prefix="bench_"))
    os.environ.setdefault("REQCLUSTER_JOB_WORKERS", "0")

    from reqcluster.db import init_db, session_scope
    from reqcluster.domain import ClusterInfo, RequirementIn
    from reqcluster.export import export
    from reqcluster.ingest import IngestResult
    from reqcluster.models import RequirementRow
    from reqcluster.services import clustering, sessions
    from sqlalchemy import select

    init_db()
    n, rng = args.n, np.random.default_rng(0)
    reqs = [RequirementIn(req_id=f"REQ-{i:05d}", ordinal=i + 1,
                          text=f"The subsystem {i % 97} shall report status {i} within {i % 50} ms",
                          attributes={"module": f"M{i % 17}", "section": f"S{i % 5}"})
            for i in range(n)]
    sid = sessions.create(IngestResult(reqs, {}), "bench").id
    with session_scope() as s:
        pks = list(s.scalars(select(RequirementRow.id).where(RequirementRow.session_id == sid)
                             .order_by(RequirementRow.ordinal)))
    labels = rng.integers(-1, 20, n)
    clusters = [ClusterInfo(cluster_id=c, size=int((labels == c).sum()), label=f"c{c}",
                            keywords=["a", "b"]) for c in range(20)]
    edges = [[pks[int(a)], pks[int(b)], 0.8] for a, b in rng.integers(0, n, (2 * n, 2)) if a != b]
    clustering.persist_run(sid, "cluster", {}, pks, labels, rng.random(n).astype(np.float32),
                           rng.random((n, 10)).astype(np.float32),
                           rng.random((n, 2)).astype(np.float32), clusters, edges, {}, [], {})

    for name, fn in (("graph_json", lambda: sessions.graph_json(sid)),
                     ("export_csv", lambda: export(sid, "csv"))):
        prof = cProfile.Profile()
        t = time.perf_counter()
        out = prof.runcall(fn)
        dt = time.perf_counter() - t
        size = len(out if isinstance(out, bytes) else out[0])
        print(f"\n== {name}: {dt:.3f}s cold, {size / 1e6:.2f} MB")
        st = pstats.Stats(prof)
        for k in sorted(st.stats, key=lambda k: st.stats[k][2], reverse=True)[:args.top]:
            print(f"  {st.stats[k][2]:7.3f}s self  x{st.stats[k][1]:<7} {k[2]} ({os.path.basename(k[0])}:{k[1]})")


if __name__ == "__main__":
    main()
