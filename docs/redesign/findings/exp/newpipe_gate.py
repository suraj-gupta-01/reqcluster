"""Accuracy + speed gate for the new reqcluster.ml.pipeline on real labeled data.

Usage: uv run python docs/redesign/findings/exp/newpipe_gate.py EMB_DIR
EMB_DIR holds (or receives) emb_<dataset>.npy (MiniLM, L2-normalised, row order = CSV order), as written
by umap06.py. Floors from 04-ml-defaults.md.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from reqcluster.domain import ClusteringParams
from reqcluster.ml import metrics, pipeline

FLOORS = {"uav_fms_2k": 0.92, "uav_fms_10k": 0.89, "multidomain": 0.62, "adversarial": 0.32}
DATA = Path(__file__).resolve().parents[4] / "data"

emb_dir = Path(sys.argv[1])
fail = False
for name, floor in FLOORS.items():
    df = pd.read_csv(DATA / f"{name}_requirements.csv", keep_default_na=False)
    f = emb_dir / f"emb_{name}.npy"
    if not f.exists():  # encode once on CPU and cache, so device float noise stays out of the gate
        from sentence_transformers import SentenceTransformer
        emb_dir.mkdir(parents=True, exist_ok=True)
        np.save(f, SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(
            df.text.astype(str).tolist(), batch_size=64, normalize_embeddings=True))
    X = np.load(f).astype(np.float32)
    texts = df.text.astype(str).tolist()
    t = time.perf_counter()
    res = pipeline.run(X, texts, ClusteringParams(), seed=42, threads=12)
    total = time.perf_counter() - t
    m = metrics.compute(res.labels, res.x10, 42, ground_truth=df.module.astype(str).values)
    ok = m["ari"] >= floor
    fail |= not ok
    print(json.dumps({"ds": name, "n": len(X), "ari": round(m["ari"], 3), "floor": floor,
                      "pass": ok, "acc_noise_incl": round(m["accuracy_noise_inclusive"], 3),
                      "noise_rate": round(m["noise_rate"], 3), "seconds": round(total, 1),
                      "steps": {k: round(v, 1) for k, v in res.durations.items()}}), flush=True)
sys.exit(1 if fail else 0)
