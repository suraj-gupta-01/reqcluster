"""Prod-path check: build_texts(base) (boilerplate stripped) + pipeline.run (unique-row clustering), CPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from reqcluster.domain import ClusteringParams
from reqcluster.embeddings import SbertEmbedder
from reqcluster.ml import pipeline
from reqcluster.texts import build_texts

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "var" / "cache" / "gate"
DS = ["aerospace", "bit_control", "sample", "uav_fms_phase2_audit", "dependency_gold",
      "uav_fms_stress_600", "adversarial", "promise", "multidomain", "uav_fms_2k"]
MODELS = {"minilm": "all-MiniLM-L6-v2", "gte": "Alibaba-NLP/gte-modernbert-base"}


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    rows = []
    for m, name in MODELS.items():
        emb = None
        for ds in DS:
            df = pd.read_csv(ROOT / "data" / f"{ds}_requirements.csv", keep_default_na=False)
            raw = df["text"].astype(str).tolist() if "text" in df else df.iloc[:, 1].astype(str).tolist()
            gold = df["module"].astype(str).tolist()
            texts = build_texts(raw, "base")[0]
            h = hashlib.sha1("\n".join(texts).encode()).hexdigest()[:10]
            f = CACHE / f"emb_{m}_prod_{ds}_{h}.npy"
            if f.exists():
                E = np.load(f)
            else:
                emb = emb or SbertEmbedder(name, "main", "cpu")
                E = np.asarray(emb.encode(texts, 32), dtype=np.float32)
                np.save(f, E)
            aris, noises = [], []
            for s in (0, 1, 2):
                lab = np.asarray(pipeline.run(E, raw, ClusteringParams(), seed=s).labels)
                aris.append(adjusted_rand_score(gold, lab))
                noises.append(float((lab < 0).mean()))
            rows.append({"ds": ds, "model": m, "ari": float(np.mean(aris)), "noise": float(np.mean(noises))})
            print(f"{ds:24s} {m:7s} ARI {rows[-1]['ari']:.3f} noise {rows[-1]['noise']:.3f}", flush=True)
    (CACHE / "prod_check.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


if __name__ == "__main__":
    main()
