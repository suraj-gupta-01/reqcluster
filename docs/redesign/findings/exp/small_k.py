"""Small uploads: does a tighter k cap fix ward+silhouette over-splitting? vs UMAP+HDBSCAN.

Full small datasets (aerospace 64, bit_control 100, dependency_gold 371) plus 5 random 60/150-row
subsets of multidomain and uav_fms_2k. Usage: uv run python docs/redesign/findings/exp/small_k.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import silhouette_score

DATA = Path(__file__).resolve().parents[4] / "data"
CAPS = {"n//3": lambda n: n // 3, "n//5": lambda n: n // 5, "sqrt(n)": lambda n: int(np.sqrt(n)),
        "sqrt(2n)": lambda n: int(np.sqrt(2 * n))}


def ward(X, kmax):
    best, best_s = None, -2.0
    for k in range(2, max(2, min(30, kmax)) + 1):
        lab = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        s = silhouette_score(X, lab, metric="cosine")
        if s > best_s:
            best, best_s = lab, s
    return best


def uh(X, seed=0):
    n = len(X)
    Z = umap.UMAP(n_components=10, metric="cosine", n_neighbors=min(15, n - 1), min_dist=0.0,
                  n_epochs=200, init="random", random_state=seed,
                  compatibility_layout=True).fit_transform(X)
    return HDBSCAN(min_cluster_size=max(5, n // 50), min_samples=3).fit_predict(Z)


model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
cases = []
for name in ("aerospace", "bit_control", "dependency_gold", "multidomain", "uav_fms_2k"):
    d = pd.read_csv(DATA / f"{name}_requirements.csv", keep_default_na=False)
    if "module" not in d:
        continue
    X = model.encode(d.text.astype(str).tolist(), batch_size=64, normalize_embeddings=True)
    y = d.module.astype(str).values
    if len(X) <= 400:
        cases.append((f"{name}:{len(X)}", [(X, y)]))
    else:
        for size in (60, 150):
            subs = [np.random.default_rng(r).choice(len(X), size, replace=False) for r in range(5)]
            cases.append((f"{name}:{size}", [(X[i], y[i]) for i in subs]))

for label, draws in cases:
    row = {"case": label, "groups": len(set(draws[0][1]))}
    row["umap_hdbscan"] = round(float(np.mean([ari(y, uh(X)) for X, y in draws])), 3)
    for cap, f in CAPS.items():
        row[f"ward {cap}"] = round(float(np.mean([ari(y, ward(X, f(len(X)))) for X, y in draws])), 3)
    print(json.dumps(row), flush=True)
