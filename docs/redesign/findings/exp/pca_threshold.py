"""Does PCA50 before UMAP help or hurt by dataset size? (MiniLM, current UMAP/HDBSCAN defaults)

Usage: uv run python docs/redesign/findings/exp/pca_threshold.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

SEEDS = (0, 1, 2)
DATA = Path(__file__).resolve().parents[4] / "data"
model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
for name in (sys.argv[1:] or ("aerospace", "adversarial", "multidomain", "uav_fms_2k")):
    d = pd.read_csv(DATA / f"{name}_requirements.csv", keep_default_na=False)
    X = model.encode(d.text.astype(str).tolist(), batch_size=64, normalize_embeddings=True)
    n = len(X)
    for pca in (True, False):
        scores, t0 = [], time.time()
        for s in SEEDS:
            P = PCA(min(50, n - 1), svd_solver="randomized", random_state=s).fit_transform(X) if pca else X
            Z = umap.UMAP(n_components=10, metric="cosine", n_neighbors=min(80, max(15, n // 100)),
                          min_dist=0.0, n_epochs=200, init="random", random_state=s,
                          compatibility_layout=True).fit_transform(P)
            lab = HDBSCAN(min_cluster_size=max(5, n // 50), min_samples=3).fit_predict(Z)
            scores.append(ari(d.module.astype(str).values, lab))
        print(json.dumps({"ds": name, "n": n, "pca": pca, "ari": round(float(np.mean(scores)), 3),
                          "std": round(float(np.std(scores)), 3),
                          "s_per_seed": round((time.time() - t0) / len(SEEDS), 1)}), flush=True)
