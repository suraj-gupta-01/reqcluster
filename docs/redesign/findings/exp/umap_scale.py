"""UMAP-internal speed options at 10k/35k: 3 seeds, ARI (hdbscan on x10 vs `module`) and seconds.

    uv run python docs/redesign/findings/exp/umap_scale.py [--seeds 3] [--sizes 10k,35k]
Embeddings: MiniLM on CPU, cached to var/cache/gate/emb_uav_fms_<size>.npy.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "backend"))
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

from reqcluster.ml import cluster, reduce  # noqa: E402


def embeddings(size: str):
    df = pd.read_csv(ROOT / "data" / f"uav_fms_{size}_requirements.csv", keep_default_na=False)
    p = ROOT / "var" / "cache" / "gate" / f"emb_uav_fms_{size}.npy"
    if not p.exists():
        from sentence_transformers import SentenceTransformer

        p.parent.mkdir(parents=True, exist_ok=True)
        st = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        x = st.encode(df.text.astype(str).tolist(), batch_size=64, normalize_embeddings=True)
        np.save(p, x.astype(np.float32))
    return np.load(p), df.module.astype(str).values


def run10(X, seed, knn, nn, epochs=200, compat=True):
    import umap

    t = time.time()
    x10 = umap.UMAP(n_components=10, metric="cosine", n_neighbors=nn, min_dist=0.0, n_epochs=epochs,
                    init="random", random_state=seed, compatibility_layout=compat,
                    precomputed_knn=knn).fit_transform(X)
    return x10, time.time() - t


def runviz(x10, seed, knn=None, epochs=100, nn=15):
    import umap

    t = time.time()
    umap.UMAP(n_components=2, metric="euclidean", n_neighbors=nn, min_dist=0.1, n_epochs=epochs,
              init="random", random_state=seed, compatibility_layout=True,
              precomputed_knn=knn if knn is not None else (None, None, None)).fit_transform(x10)
    return time.time() - t


def score(x10, y):
    return ari(y, cluster.hdbscan(x10, cluster.auto_min_cluster_size(len(y)), 3)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sizes", default="10k,35k")
    a = ap.parse_args()
    rows = []
    Xw = embeddings(a.sizes.split(",")[0])[0][:1500]  # JIT warm-up so first timings are not compile time
    kw = reduce._exact_knn(Xw, 15)
    for c in (True, False):
        w, _ = run10(Xw, 0, kw, 15, 20, c)
    runviz(w, 0, epochs=20)
    runviz(w, 0, (kw[0], kw[1], None), epochs=20)
    for size in a.sizes.split(","):
        X, y = embeddings(size)
        n = len(X)
        nn = min(80, max(15, n // 100), n - 1)
        t = time.time()
        knn = reduce._exact_knn(X, nn)
        print(f"{size}: n={n} nn={nn} exact kNN {time.time() - t:.1f}s", flush=True)
        for seed in range(a.seeds):
            # 10-D variants: (name, epochs, compat)
            for name, ep, compat in (("base e200", 200, True), ("e100", 100, True), ("e140", 140, True),
                                     ("adam e200", 200, False), ("adam e100", 100, False)):
                x10, s = run10(X, seed, knn, nn, ep, compat)
                rows.append((size, name, seed, s, score(x10, y)))
                print(rows[-1], flush=True)
                if name == "base e200":
                    base = x10
            # viz variants on base x10 (ARI unaffected): seconds only
            k15 = (np.ascontiguousarray(knn[0][:, :15]), np.ascontiguousarray(knn[1][:, :15]), None)
            for name, kw in (("viz e100", {}), ("viz e50", {"epochs": 50}), ("viz e30", {"epochs": 30}),
                             ("viz e50 knn", {"epochs": 50, "knn": k15})):
                rows.append((size, name, seed, runviz(base, seed, **kw), float("nan")))
                print(rows[-1], flush=True)
    df = pd.DataFrame(rows, columns=["size", "option", "seed", "sec", "ari"])
    print(df.groupby(["size", "option"], sort=False).agg(
        sec=("sec", "mean"), ari=("ari", "mean"), ari_std=("ari", "std")).round(3).to_string())


if __name__ == "__main__":
    main()
