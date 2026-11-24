"""Ceiling vs production score per feature set (which features can reach high ARI).

Per (feature, dataset): ceil_lr (5-fold LogisticRegression ARI/acc), ward_k (oracle k Ward ARI),
prod (small-upload path: 7 UMAP seeds, consensus of 5, noise = one label; mean over seeds 0-2).
Run: uv run python docs/redesign/findings/exp/ceiling_features.py
"""
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import adjusted_rand_score as ari  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "var" / "cache" / "gate"
NAMES = ["aerospace", "bit_control", "sample", "uav_fms_phase2_audit", "dependency_gold",
         "uav_fms_stress_600", "adversarial", "promise", "multidomain", "uav_fms_2k"]
INSTR = "Instruct: Identify the subsystem or functional area this system requirement belongs to\nQuery: "
# feat -> (model, prompt, kwargs)
MODELS = {
    "minilm": ("all-MiniLM-L6-v2", None, {}),
    "gte": ("Alibaba-NLP/gte-modernbert-base", None, {}),
    "qwen3": ("Qwen/Qwen3-Embedding-0.6B", None, {"model_kwargs": {"torch_dtype": "float16"}}),
    "qwen3i": ("Qwen/Qwen3-Embedding-0.6B", INSTR, {"model_kwargs": {"torch_dtype": "float16"}}),
    "bgem3": ("BAAI/bge-m3", None, {}),
    "mxbai": ("mixedbread-ai/mxbai-embed-large-v1", None, {}),
    "mpnet": ("sentence-transformers/all-mpnet-base-v2", None, {}),
    "e5": ("intfloat/e5-base-v2", "query: ", {}),
}
HYB = ("minilm", "gte", "qwen3i")


def load(name):
    d = pd.read_csv(ROOT / "data" / f"{name}_requirements.csv", keep_default_na=False)
    return d["text"].astype(str).tolist(), d["module"].astype(str).values


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def feat_path(f, ds):
    return CACHE / f"emb_{f}_{ds}.npy"


def embed_all(failed):
    import torch
    from sentence_transformers import SentenceTransformer

    texts = {ds: load(ds)[0] for ds in NAMES}
    for f, (mid, prompt, kw) in MODELS.items():
        todo = [ds for ds in NAMES if not feat_path(f, ds).exists()]
        if not todo:
            continue
        try:
            kw = {k: ({"torch_dtype": torch.float16} if k == "model_kwargs" else v) for k, v in kw.items()}
            m = SentenceTransformer(mid, device="cuda", trust_remote_code=True, **kw)
            for ds in todo:
                E = m.encode(texts[ds], batch_size=16, prompt=prompt, show_progress_bar=False)
                np.save(feat_path(f, ds), l2(np.asarray(E, dtype=np.float32)))
            del m
        except Exception as e:  # noqa: BLE001
            failed[f] = repr(e)[:100]
        torch.cuda.empty_cache()


def make_tfidf_hyb(failed):
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    for ds in NAMES:
        if not feat_path("tfidf", ds).exists():
            X = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), min_df=2,
                                stop_words="english").fit_transform(load(ds)[0])
            T = TruncatedSVD(min(100, X.shape[1] - 1), random_state=0).fit_transform(X)
            np.save(feat_path("tfidf", ds), l2(T).astype(np.float32))
        T = np.load(feat_path("tfidf", ds))
        for e in HYB:
            if feat_path(e, ds).exists():
                np.save(feat_path(f"hyb_{e}", ds), l2(np.hstack([l2(np.load(feat_path(e, ds))), 0.5 * T])))


def task(a):
    f, ds = a
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from reqcluster.ml import cluster, reduce

    X = l2(np.load(feat_path(f, ds)).astype(np.float32))
    y = load(ds)[1]
    n = len(y)
    k = len(set(y))
    cv = (StratifiedKFold if np.unique(y, return_counts=True)[1].min() >= 5 else KFold)(
        5, shuffle=True, random_state=0)
    p = cross_val_predict(LogisticRegression(C=10, max_iter=2000), X, y, cv=cv)
    row = {"feat": f, "ds": ds, "ceil_ari": ari(y, p), "ceil_acc": float((p == y).mean())}
    row["ward_k"] = ari(y, AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X))
    mcs = cluster.auto_min_cluster_size(n)
    runs = [cluster.hdbscan(reduce.reduce(X, s)[0], mcs, 3)[0] for s in range(7)]
    aris, noise = [], []
    for s in range(3):
        lab = cluster.consensus(runs[s:s + 5], mcs, 3, X)[0] if 20 <= n <= 2000 else runs[s]
        lab = np.asarray(lab)
        aris.append(ari(y, lab))
        noise.append(float((lab == -1).mean()))
    row["prod_ari"], row["prod_noise"] = float(np.mean(aris)), float(np.mean(noise))
    return row


def main():
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    failed = {}
    embed_all(failed)
    make_tfidf_hyb(failed)
    feats = [f for f in [*MODELS, "tfidf", *(f"hyb_{e}" for e in HYB)]
             if all(feat_path(f, ds).exists() for ds in NAMES)]
    jobs = [(f, ds) for ds in reversed(NAMES) for f in feats]
    with get_context("spawn").Pool(8) as pool:
        rows = pool.map(task, jobs, chunksize=1)
    with open(CACHE / "ceiling_features.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    for col in ("ceil_ari", "prod_ari"):
        print(col)
        print(df.pivot(index="ds", columns="feat", values=col)[feats].reindex(NAMES).round(3).to_string())
    for f in feats:
        print(f"ward_k {f}: {df[df.feat == f].ward_k.mean():.3f}")
    print("failed:", failed or "none", f"| {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
