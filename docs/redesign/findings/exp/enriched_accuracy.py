"""Does LLM-enriched text raise clustering ARI on small uploads?

Run: uv run python docs/redesign/findings/exp/enriched_accuracy.py
Needs a local Ollama with the default model (qwen3.5:4b). Enrichment output is
cached in var/cache/gate/enriched_<name>.json and reused on later runs; delete
it to regenerate. Compares base / enriched / hybrid texts, seeds 0-9, through
the production pipeline.run (5-seed consensus inside), scored by ARI vs module
(noise = one label).
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ["REQCLUSTER_LLM_PROVIDER"] = "local"

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from reqcluster.domain import ClusteringParams
from reqcluster.embeddings import SbertEmbedder
from reqcluster.llm.enrich import expand_all
from reqcluster.llm.gateway import get_gateway
from reqcluster.ml import pipeline
from reqcluster.texts import build_texts

ROOT = Path(__file__).resolve().parents[4]
NAMES = ["aerospace", "bit_control", "sample", "uav_fms_phase2_audit"]
MODES = ["base", "enriched", "hybrid"]
SEEDS = range(10)
CACHE = ROOT / "var" / "cache" / "gate"


def load_enriched(name, texts):
    path = CACHE / f"enriched_{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8")), None
    t0 = time.time()
    res = expand_all([(i, t) for i, t in enumerate(texts)], get_gateway(), "local")
    per = (time.time() - t0) / len(texts)
    rows = [{"i": r.requirement_id, "status": r.status, "source": r.source,
             "expanded_text": r.expanded_text} for r in res]
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows), encoding="utf-8")
    os.replace(tmp, path)
    return rows, per


def main():
    emb = SbertEmbedder("all-MiniLM-L6-v2", "main", "cpu")
    table = []
    for name in NAMES:
        df = pd.read_csv(ROOT / "data" / f"{name}_requirements.csv", keep_default_na=False)
        texts, module = df["text"].tolist(), df["module"].tolist()
        rows, per = load_enriched(name, texts)
        bad = [r for r in rows if r["source"] == "fallback" or r["status"] != "succeeded"]
        failed = sum(r["status"] != "succeeded" for r in rows)
        print(f"{name}: n={len(texts)} fallback/failed={len(bad)} failed={failed} "
              f"llm_s_per_req={'cached' if per is None else f'{per:.2f}'}", flush=True)
        enriched = [None if r in bad else r["expanded_text"] for r in rows]
        for mode in MODES:
            mt, _ = build_texts(texts, mode, enriched)
            X = emb.encode(mt, 32)
            aris, noise = [], []
            for s in SEEDS:
                labels = np.asarray(pipeline.run(X, texts, ClusteringParams(), seed=s).labels)
                aris.append(adjusted_rand_score(module, labels))
                noise.append(float((labels == -1).mean()))
            table.append((name, mode, np.mean(aris), np.std(aris), np.min(aris), np.mean(noise)))
            print(f"{name:22s} {mode:9s} ARI {table[-1][2]:.3f} sd {table[-1][3]:.3f} "
                  f"min {table[-1][4]:.3f} noise {table[-1][5]:.3f}", flush=True)
    print("\noverall means")
    for mode in MODES:
        rs = [r for r in table if r[1] == mode]
        print(f"{mode:9s} ARI {np.mean([r[2] for r in rs]):.3f} noise {np.mean([r[5] for r in rs]):.3f}")


if __name__ == "__main__":
    sys.exit(main())
