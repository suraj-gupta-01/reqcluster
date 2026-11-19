"""hdbscan vs ward (no-noise mode): mean ARI and noise-inclusive accuracy, real pipeline, MiniLM cache.

Run: uv run python docs/redesign/findings/exp/no_noise_mode.py
Accuracy: Hungarian match of gold modules to clusters; gold '__noise__' rows count correct iff label -1.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score as ari

from reqcluster.domain import ClusteringParams
from reqcluster.ml import pipeline

ROOT = Path(__file__).resolve().parents[4]
NAMES = ["uav_fms_2k", "uav_fms_10k", "multidomain", "adversarial"]


def accuracy(gold, pred):
    isn = gold == "__noise__"
    ok = int((isn & (pred < 0)).sum())
    gl, pl = sorted(set(gold[~isn])), sorted(set(pred[pred >= 0]))
    if pl:
        M = np.zeros((len(gl), len(pl)), int)
        gi, pi = {g: i for i, g in enumerate(gl)}, {p: i for i, p in enumerate(pl)}
        for g, p in zip(gold[~isn], pred[~isn], strict=True):
            if p >= 0:
                M[gi[g], pi[p]] += 1
        r, c = linear_sum_assignment(-M)
        ok += int(M[r, c].sum())
    return ok / len(gold)


for name in NAMES:
    df = pd.read_csv(ROOT / "data" / f"{name}_requirements.csv", keep_default_na=False)
    X = np.load(ROOT / "var" / "cache" / "gate" / f"emb_{name}.npy")
    gold, texts = df["module"].astype(str).values, df["text"].astype(str).tolist()
    for method in ("hdbscan", "ward"):
        a, c = [], []
        for seed in (0, 1, 2):
            lab = pipeline.run(X, texts, ClusteringParams(clusterer=method), seed=seed).labels
            a.append(ari(gold, lab))
            c.append(accuracy(gold, lab))
        print(f"{name:12s} {method:8s} ARI={np.mean(a):.3f} acc={np.mean(c):.3f}", flush=True)
