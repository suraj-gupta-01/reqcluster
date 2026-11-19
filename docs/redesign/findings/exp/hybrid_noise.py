"""Can ward's better grouping keep outlier detection? Compare (3 seeds, MiniLM, real reduce):

  hdbscan      default: HDBSCAN on UMAP 10-D (finds noise)
  ward         no-noise mode
  ward+hdb     ward groups; rows HDBSCAN calls noise become noise
  ward+mad     ward groups; a row is noise when its cosine to its own cluster centroid (raw embeddings)
               is below that cluster's median - 3 * MAD (cluster-relative outliers, no global threshold)

Reports ARI (noise as its own label), noise-inclusive accuracy, predicted noise rate, and on
`adversarial` the precision/recall of noise against its 120 `__noise__` rows.
    uv run python docs/redesign/findings/exp/hybrid_noise.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reqcluster.ml import cluster, metrics, reduce

ROOT = Path(__file__).resolve().parents[4]
GATE = ROOT / "var" / "cache" / "gate"


def mad_noise(X, labels, k=3.0):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    out = labels.copy()
    for c in np.unique(labels[labels >= 0]):
        idx = np.flatnonzero(labels == c)
        cen = Xn[idx].mean(0)
        sim = Xn[idx] @ (cen / np.linalg.norm(cen))
        med = np.median(sim)
        mad = np.median(np.abs(sim - med)) or 1e-9
        out[idx[sim < med - k * 1.4826 * mad]] = -1
    return out


rows = []
for ds in ("uav_fms_2k", "uav_fms_10k", "multidomain", "adversarial"):
    X = np.load(GATE / f"emb_{ds}.npy").astype(np.float32)
    y = pd.read_csv(ROOT / "data" / f"{ds}_requirements.csv", keep_default_na=False).module.astype(str).values
    true_noise = y == "__noise__"
    res: dict[str, list] = {}
    for seed in (0, 1, 2):
        x10, _ = reduce.reduce(X, seed)
        hdb = cluster.hdbscan(x10, cluster.auto_min_cluster_size(len(X)), 3)[0]
        w = cluster.ward(x10, seed)[0]
        variants = {"hdbscan": hdb, "ward": w,
                    "ward+hdb": np.where(hdb == -1, -1, w), "ward+mad": mad_noise(X, w)}
        for name, lab in variants.items():
            m = metrics.compute(lab, None, seed, ground_truth=y)
            r = [m["ari"], m["accuracy_noise_inclusive"], float((lab == -1).mean())]
            if true_noise.any():
                pred = lab == -1
                tp = float((pred & true_noise).sum())
                r += [tp / max(pred.sum(), 1), tp / true_noise.sum()]
            res.setdefault(name, []).append(r)
    for name, vals in res.items():
        v = np.mean(vals, axis=0)
        row = {"ds": ds, "variant": name, "ari": round(v[0], 3), "acc_incl": round(v[1], 3),
               "noise_rate": round(v[2], 3)}
        if len(v) > 3:
            row |= {"noise_precision": round(v[3], 3), "noise_recall": round(v[4], 3)}
        print(json.dumps(row), flush=True)
