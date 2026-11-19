"""Profile one call and show the exact code that costs the time.

Caveat: cProfile charges numba JIT compilation to whatever Python frame runs the jitted code
first, so on a cold process it can blame the wrong function. Trust the wall-clock step timings
printed first; use --warm to profile after JIT so the call graph reflects steady-state cost.

Prints the hot path (from the entry point, follow the callee with the most cumulative time) with
% of total and file:line at each step, and writes a Mermaid call graph of every function above
--min-pct (open the .mmd in VS Code / GitHub to see it).

    uv run python scripts/profile_graph.py pipeline --n 40 --warm   # clustering pipeline, N points
    uv run python scripts/profile_graph.py expr "import numpy; numpy.linalg.svd(numpy.ones((500,500)))"
"""
import argparse
import cProfile
import pstats
from pathlib import Path


def _label(fn) -> str:
    file, line, name = fn
    p = Path(file)
    short = "/".join(p.parts[-2:]) if p.parts else file
    return f"{name} ({short}:{line})"


def _targets(args):
    if args.target == "expr":
        return lambda: exec(args.code, {})  # noqa: S102 - developer tool, code comes from argv
    import numpy as np

    from reqcluster.domain import ClusteringParams
    from reqcluster.ml import pipeline

    rng = np.random.default_rng(0)
    centers = rng.standard_normal((4, args.dim))
    X = centers[rng.integers(0, 4, args.n)] + 0.3 * rng.standard_normal((args.n, args.dim))
    X = (X / np.linalg.norm(X, axis=1, keepdims=True)).astype(np.float32)
    texts = [f"requirement {i}" for i in range(args.n)]
    def run():
        res = pipeline.run(X, texts, ClusteringParams(), seed=42, threads=args.threads)
        print("wall-clock steps:", {k: round(v, 3) for k, v in res.durations.items()})
        return res

    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("target", choices=["pipeline", "expr"])
    ap.add_argument("code", nargs="?", default="", help="python code for target=expr")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dim", type=int, default=384)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--warm", action="store_true", help="run once before profiling (excludes JIT)")
    ap.add_argument("--min-pct", type=float, default=2.0, help="graph: hide functions below this")
    ap.add_argument("--out", default="var/reports/profile_graph.mmd")
    args = ap.parse_args()

    fn = _targets(args)
    if args.warm:
        fn()
    prof = cProfile.Profile()
    prof.runcall(fn)
    st = pstats.Stats(prof)
    stats = st.stats  # {func: (cc, nc, tt, ct, callers{caller: (cc, nc, tt, ct)})}
    total = max(v[3] for v in stats.values())
    callees: dict = {}
    for fn_key, (*_, callers) in stats.items():
        for caller, (_, _, _, ct) in callers.items():
            # cProfile can record 0 on edges into numba/C code; use the callee's own total then
            callees.setdefault(caller, []).append((ct or stats[fn_key][3], fn_key))

    root = max(stats, key=lambda k: stats[k][3])
    print(f"total {total:.2f}s\nHOT PATH (cumulative time, % of total, self time):")
    node, seen, depth = root, set(), 0
    while node and node not in seen and depth < 40:
        seen.add(node)
        _, nc, tt, ct, _ = stats[node]
        print(f"{'  ' * min(depth, 12)}{ct:8.2f}s {100 * ct / total:5.1f}%  self {tt:6.2f}s  "
              f"x{nc}  {_label(node)}")
        kids = sorted(callees.get(node, []), reverse=True)
        node = kids[0][1] if kids and kids[0][0] >= 0.01 * total else None
        depth += 1

    print("\nTOP SELF TIME (where the CPU actually is):")
    for k in sorted(stats, key=lambda k: stats[k][2], reverse=True)[:12]:
        print(f"  {stats[k][2]:8.2f}s  {100 * stats[k][2] / total:5.1f}%  {_label(k)}")

    keep = {k for k, v in stats.items() if 100 * v[3] / total >= args.min_pct}
    ids = {k: f"f{i}" for i, k in enumerate(keep)}
    lines = ["flowchart TD"]
    for k in keep:
        name = _label(k).replace('"', "'")
        lines.append(f'  {ids[k]}["{name}<br/>{stats[k][3]:.2f}s ({100 * stats[k][3] / total:.0f}%)"]')
    for caller, kids in callees.items():
        for ct, k in kids:
            if caller in keep and k in keep and 100 * ct / total >= args.min_pct:
                lines.append(f"  {ids[caller]} -->|{ct:.2f}s| {ids[k]}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\ncall graph ({len(keep)} functions >= {args.min_pct}%): {out}")


if __name__ == "__main__":
    main()
