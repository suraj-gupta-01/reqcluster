"""Small uploads (60-300 rows): UMAP+HDBSCAN vs deterministic agglomerative clustering.

For each size, draw 5 random subsets of real labeled datasets (MiniLM embeddings) and score ARI.
`stab` = ARI between the labels on X and on X + 1e-6 noise (a CPU-vs-GPU sized perturbation);
1.0 means the method is insensitive to it.
Usage: uv run python docs/redesign/findings/exp/small_n.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score

DATA = Path(__file__).resolve().parents[4] / "data"


def umap_hdbscan(X, seed):
    n = len(X)
    Z = umap.UMAP(n_components=min(10, n - 2), metric="cosine", n_neighbors=min(15, n - 1),
                  min_dist=0.0, n_epochs=200, init="random", random_state=seed,
                  compatibility_layout=True).fit_transform(X)
    return HDBSCAN(min_cluster_size=max(5, n // 50), min_samples=3).fit_predict(Z)


def agglo(X, linkage):
    n = len(X)
    best, best_s = None, -2.0
    for k in range(2, max(3, min(30, n // 3)) + 1):
        kw = {"metric": "cosine"} if linkage == "average" else {}
        lab = AgglomerativeClustering(n_clusters=k, linkage=linkage, **kw).fit_predict(X)
        s = silhouette_score(X, lab, metric="cosine")
        if s > best_s:
            best, best_s = lab, s
    return best


METHODS = {"umap_hdbscan": lambda X, s: umap_hdbscan(X, s),
           "agglo_average": lambda X, s: agglo(X, "average"),
           "agglo_ward": lambda X, s: agglo(X, "ward")}

model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
pools = {}
for name in ("aerospace", "multidomain", "uav_fms_2k", "adversarial"):
    d = pd.read_csv(DATA / f"{name}_requirements.csv", keep_default_na=False)
    pools[name] = (model.encode(d.text.astype(str).tolist(), batch_size=64,
                                normalize_embeddings=True), d.module.astype(str).values)

for size in (int(a) for a in (sys.argv[1:] or (60, 150, 300))):
    for name, (X_all, y_all) in pools.items():
        if len(X_all) < size:
            continue
        for m, fn in METHODS.items():
            scores, stab = [], []
            for draw in range(5):
                idx = np.random.default_rng(draw).choice(len(X_all), size, replace=False)
                X, y = X_all[idx], y_all[idx]
                lab = fn(X, draw)
                scores.append(ari(y, lab))
                Xp = X + np.random.default_rng(99).normal(0, 1e-6, X.shape).astype(X.dtype)
                stab.append(ari(lab, fn(Xp, draw)))
            print(json.dumps({"n": size, "ds": name, "method": m, "ari": round(float(np.mean(scores)), 3),
                              "std": round(float(np.std(scores)), 3),
                              "stab": round(float(np.mean(stab)), 3)}), flush=True)
