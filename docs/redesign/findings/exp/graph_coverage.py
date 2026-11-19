"""R2-S4: graph edge coverage on uav_fms_10k, old (global top-N) vs new (per-node best 2 first)."""
import csv
from pathlib import Path

import numpy as np

from reqcluster.domain import ClusteringParams
from reqcluster.ml import graph, pipeline, sampling

ROOT = Path(__file__).resolve().parents[4]
CSV = ROOT / "data" / "uav_fms_10k_requirements.csv"
NPY = ROOT / "var" / "cache" / "gate" / "emb_uav_fms_10k.npy"
THRESHOLD, K, MAX_EDGES = 0.5, 10, 4000

with CSV.open(encoding="utf-8", newline="") as f:
    texts = [r["text"] for r in csv.DictReader(f)]
if NPY.exists():
    X = np.load(NPY)
else:
    from sentence_transformers import SentenceTransformer
    X = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(texts, batch_size=64, normalize_embeddings=True)
    NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(NPY, X)

res = pipeline.run(X, texts, ClusteringParams(), seed=42)
labels = np.asarray(res.labels)
nodes = np.flatnonzero(labels != -1)


def old(X, labels, threshold, k, max_edges):
    Xn = np.asarray(X, dtype=np.float32)[nodes]
    Xn = Xn / np.maximum(np.linalg.norm(Xn, axis=1, keepdims=True), 1e-12)
    res = graph._hnsw_edges(Xn, k, 0, 1)
    nb, sim = res if res is not None else sampling.knn(Xn, k)
    best = {}
    for li in range(len(nodes)):
        for lj, s in zip(nb[li], sim[li], strict=True):
            if int(lj) == li or s < threshold:
                continue
            key = tuple(sorted((int(nodes[li]), int(nodes[int(lj)]))))
            best[key] = max(best.get(key, -1.0), float(s))
    e = [(a, b, round(s, 4)) for (a, b), s in best.items()]
    e.sort(key=lambda t: (-t[2], t[0], t[1]))
    return e[:max_edges]


print(f"nodes (non-noise): {len(nodes)} of {len(labels)}")
for name, edges in (("before", old(X, labels, THRESHOLD, K, MAX_EDGES)),
                    ("after", graph.build(X, labels, THRESHOLD, K, 0, 1))):
    w = np.array([e[2] for e in edges])
    covered = len({i for e in edges for i in e[:2]})
    print(f"{name}: edges={len(edges)} nodes_with_edge={covered} min_w={w.min():.4f} median_w={np.median(w):.4f}")
