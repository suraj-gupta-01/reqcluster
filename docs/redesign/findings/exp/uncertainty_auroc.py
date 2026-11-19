"""AUROC of uncertainty scores for flagging gold-misassigned points: old (1 - prob) vs centroid margin.

Usage: uv run python docs/redesign/findings/exp/uncertainty_auroc.py
Needs var/cache/gate/emb_<name>.npy (see newpipe_gate.py). Misassigned = noise, or cluster's
majority gold label differs from the point's own.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from reqcluster.domain import ClusteringParams
from reqcluster.ml import active, pipeline

ROOT = Path(__file__).resolve().parents[4]
for name in ("uav_fms_2k", "multidomain", "adversarial"):
    df = pd.read_csv(ROOT / "data" / f"{name}_requirements.csv", keep_default_na=False)
    X = np.load(ROOT / "var/cache/gate" / f"emb_{name}.npy").astype(np.float32)
    res = pipeline.run(X, df.text.astype(str).tolist(), ClusteringParams(), seed=42, threads=4)
    lab, gold = res.labels, df.module.astype(str).values
    major = {c: pd.Series(gold[lab == c]).mode()[0] for c in set(lab.tolist()) if c != -1}
    bad = np.array([k == -1 or major[k] != g for k, g in zip(lab, gold, strict=True)])
    old = np.where(lab == -1, 1.0, 1.0 - res.probabilities)
    ranked = active.uncertainty(res.probabilities, X, lab, len(lab))
    new = np.zeros(len(lab))
    for d in ranked:
        new[d["index"]] = d["uncertainty"]
    print(f"{name}: n={len(lab)} misassigned={bad.mean():.3f} "
          f"old={roc_auc_score(bad, old):.3f} new={roc_auc_score(bad, new):.3f}")
