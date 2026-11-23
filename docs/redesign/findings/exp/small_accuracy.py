"""Small uploads (64-969 rows): candidate fixes for low, seed-unstable ARI (HANDOFF open work 2).

Production reduce + HDBSCAN on 8 labelled sets, 10 seeds, both default embedders (MiniLM on
CPU, gte-modernbert on GPU). Noise-inclusive ARI (noise = one predicted label) and noise rate.
Variants:
  base     reduce(seed) -> hdbscan(auto mcs, ms=3)                       (current default)
  ms1      min_samples=1
  rescue   base, then a noise row joins its nearest cluster (cosine to centroid, full space)
           only if it is at least as close as that cluster's farthest member
  cons5    co-association over seeds s..s+4 (same mcs/ms), HDBSCAN on 1 - co-association
  vote5    same co-association; link pairs together in >= 3 of 5 runs, components < mcs = noise
  avg5     same co-association; average linkage cut at 0.5, clusters < mcs = noise
  cat3     hdbscan on the concatenated 10-D reductions of seeds s..s+2
Run: uv run python docs/redesign/findings/exp/small_accuracy.py [model ...]
"""
import functools
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.cluster import HDBSCAN  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "var" / "cache" / "gate"
NAMES = ["aerospace", "bit_control", "sample", "uav_fms_phase2_audit", "dependency_gold",
         "uav_fms_stress_600", "adversarial", "promise"]
MODELS = {"minilm": "all-MiniLM-L6-v2", "gte": "Alibaba-NLP/gte-modernbert-base"}
SEEDS = int(os.environ.get("SEEDS", 10))
NAMES = os.environ["DS"].split(",") if os.environ.get("DS") else NAMES
CONS = 5


def load(name):
    d = pd.read_csv(ROOT / "data" / f"{name}_requirements.csv", keep_default_na=False)
    return d["text"].astype(str).tolist(), d["module"].astype(str).values


@functools.lru_cache(maxsize=None)
def _embedder(model):
    from reqcluster.embeddings import SbertEmbedder

    return SbertEmbedder(MODELS[model], "main", "cuda" if model == "gte" else "cpu")


def embeddings(model, name, texts):
    f = CACHE / f"emb_{model}_{name}.npy"
    if f.exists():
        return np.load(f)
    E = _embedder(model).encode(texts, 32)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, E)
    return E


def rescue(X, lab):
    lab = lab.copy()
    ids = sorted(set(lab[lab >= 0].tolist()))
    if not ids or (lab >= 0).all():
        return lab
    Xu = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = np.stack([Xu[lab == c].mean(0) for c in ids])
    C /= np.linalg.norm(C, axis=1, keepdims=True)
    radius = np.array([(Xu[lab == c] @ C[i]).min() for i, c in enumerate(ids)])
    noise = np.where(lab == -1)[0]
    S = Xu[noise] @ C.T
    best = S.argmax(1)
    ok = S[np.arange(len(noise)), best] >= radius[best]
    lab[noise[ok]] = np.array(ids)[best[ok]]
    return lab


def coassoc(labs):
    n = len(labs[0])
    M = np.zeros((n, n))
    for lab in labs:
        same = (lab[:, None] == lab[None, :]) & (lab[:, None] >= 0)
        M += same
    return 1.0 - M / len(labs)


def _small_to_noise(lab, mcs):
    ids, cnt = np.unique(lab, return_counts=True)
    small = set(ids[cnt < mcs].tolist())
    lab = np.array([-1 if v in small else v for v in lab])
    return relabel(lab)


def relabel(lab):
    from reqcluster.ml import labels as L

    return L.relabel_contiguous(lab)


def vote(D, mcs):
    """Evidence accumulation: link pairs together in a majority of runs; components < mcs = noise."""
    from scipy.sparse.csgraph import connected_components

    _, lab = connected_components(D < 0.5, directed=False)
    return _small_to_noise(lab, mcs)


def avg_link(D, mcs):
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    Zl = linkage(squareform(D, checks=False), "average")
    return _small_to_noise(fcluster(Zl, 0.5, "distance"), mcs)


def task(args):
    model, name = args
    from reqcluster.ml import cluster, reduce

    texts, gold = load(name)
    X = embeddings(model, name, texts)
    n = len(X)
    mcs = cluster.auto_min_cluster_size(n)
    zf = CACHE / f"z_{model}_{name}.npy"
    if zf.exists():
        Z = list(np.load(zf))
    else:
        Z = [reduce.reduce(X, s)[0] for s in range(SEEDS + CONS - 1)]
        np.save(zf, np.stack(Z))
    base = [cluster.hdbscan(z, mcs, 3)[0] for z in Z]
    rows = []
    for s in range(SEEDS):
        variants = {
            "base": base[s],
            "ms1": cluster.hdbscan(Z[s], mcs, 1)[0],
            "rescue": rescue(X, base[s]),
            "cons5": HDBSCAN(min_cluster_size=mcs, min_samples=3, metric="precomputed")
            .fit_predict(coassoc(base[s:s + CONS])),
            "cat3": cluster.hdbscan(np.hstack(Z[s:s + 3]), mcs, 3)[0],
        }
        D = coassoc(base[s:s + CONS])
        variants["vote5"] = vote(D, mcs)
        variants["avg5"] = avg_link(D, mcs)
        variants["cons5+rescue"] = rescue(X, variants["cons5"])
        variants["vote5+rescue"] = rescue(X, variants["vote5"])
        variants["avg5+rescue"] = rescue(X, variants["avg5"])
        variants["ms1+rescue"] = rescue(X, variants["ms1"])
        for v, lab in variants.items():
            rows.append({"model": model, "ds": name, "n": n, "seed": s, "variant": v,
                         "ari": ari(gold, lab), "noise": float((lab == -1).mean()),
                         "k": int(len(set(lab[lab >= 0].tolist())))})
    return rows


if __name__ == "__main__":
    models = sys.argv[1:] or list(MODELS)
    for m in models:  # embed serially (GPU), then reduce in parallel
        for nm in NAMES:
            embeddings(m, nm, load(nm)[0])
    out = CACHE / f"small_accuracy{os.environ.get('TAG', '')}.jsonl"
    rows = []
    with Pool(int(os.environ.get("NP", 8))) as p:
        for r in p.imap_unordered(task, [(m, nm) for m in models for nm in NAMES]):
            rows += r
            print(r[0]["model"], r[0]["ds"], "done", flush=True)
    out.write_text("\n".join(json.dumps(r) for r in rows))
    df = pd.DataFrame(rows)
    g = df.groupby(["model", "ds", "variant"]).agg(ari=("ari", "mean"), sd=("ari", "std"),
                                                    lo=("ari", "min"), noise=("noise", "mean"))
    pd.set_option("display.width", 200, "display.max_rows", 500)
    print(g.round(3).unstack("variant").to_string())
    print(df.groupby(["model", "variant"]).agg(ari=("ari", "mean"), sd=("ari", "std"),
                                               lo=("ari", "min"), noise=("noise", "mean")).round(3))
