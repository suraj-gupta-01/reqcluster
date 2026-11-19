"""Smoke-test transformers-5-native embedders against MiniLM with the production pipeline.

ARI = mean over seeds 0,1,2 of reqcluster.ml.pipeline.run (UMAP 10D + HDBSCAN, current defaults).
Embeddings cached in var/cache/gate/emb_<tag>_<dataset>.npy. Encoding on CPU for comparable speed.
    uv run python docs/redesign/findings/exp/embedder_candidates.py [tag ...]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics import adjusted_rand_score as ari

from reqcluster.domain import ClusteringParams
from reqcluster.ml import pipeline

ROOT = Path(__file__).resolve().parents[4]
GATE = ROOT / "var" / "cache" / "gate"
DATASETS = ("multidomain", "adversarial", "uav_fms_2k")
MODELS = {  # tag: (model id, text prefix)
    "minilm": ("all-MiniLM-L6-v2", ""),
    "modernbert-nomic": ("nomic-ai/modernbert-embed-base", "clustering: "),
    "gte-modernbert": ("Alibaba-NLP/gte-modernbert-base", ""),
    "bge-m3": ("BAAI/bge-m3", ""),
}

for tag in sys.argv[1:] or MODELS:
    name, prefix = MODELS[tag]
    try:
        model = SentenceTransformer(name, device="cpu")
    except Exception as e:  # report and continue with the next candidate
        print(json.dumps({"model": tag, "error": f"{type(e).__name__}: {e}"[:300]}), flush=True)
        continue
    row = {"model": tag}
    for ds in DATASETS:
        df = pd.read_csv(ROOT / "data" / f"{ds}_requirements.csv", keep_default_na=False)
        texts, y = df.text.astype(str).tolist(), df.module.astype(str).values
        f = GATE / ("emb_" + ds + ".npy" if tag == "minilm" else f"emb_{tag}_{ds}.npy")
        if f.exists():
            X = np.load(f)
        else:
            t = time.perf_counter()
            X = model.encode([prefix + t_ for t_ in texts], batch_size=32, normalize_embeddings=True)
            row[f"{ds}_texts_per_s"] = round(len(texts) / (time.perf_counter() - t), 1)
            GATE.mkdir(parents=True, exist_ok=True)
            np.save(f, X)
        scores = [ari(y, pipeline.run(X.astype(np.float32), texts, ClusteringParams(), seed=s).labels)
                  for s in (0, 1, 2)]
        row[ds] = f"{np.mean(scores):.3f}±{np.std(scores):.3f}"
    if tag == "minilm":  # baseline speed on a fixed sample
        t = time.perf_counter()
        model.encode(texts[:1000], batch_size=32, normalize_embeddings=True)
        row["uav_fms_2k_texts_per_s"] = round(1000 / (time.perf_counter() - t), 1)
    print(json.dumps(row), flush=True)
