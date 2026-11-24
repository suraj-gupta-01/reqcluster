"""Strip templated boilerplate sentences, re-embed, compare prod/ceil ARI before vs after."""
import gc
import json
import math
import re
from collections import Counter
from multiprocessing import get_context

import numpy as np
import pandas as pd
from ceiling_features import CACHE, MODELS, NAMES, ari, feat_path, l2, load

FEATS = ("minilm", "gte", "qwen3i")


def key(s):
    return re.sub(r"\d+", "#", s.lower()).strip(" \t\n.,;:!?()[]-")


def split(t):
    return [s for s in re.split(r"(?<=[.!?;)])\s+", t) if s.strip()]


def strip(texts):
    """Return (stripped texts, number of boilerplate keys)."""
    rows = [split(t) for t in texts]
    cnt = Counter(k for r in rows for k in {key(s) for s in r})
    thr = max(3, math.ceil(0.02 * len(texts)))
    bp = {k for k, c in cnt.items() if c >= thr}
    out = []
    for t, r in zip(texts, rows, strict=True):
        kept = [s for s in r if key(s) not in bp]
        out.append(" ".join(kept) if kept else t)
    return out, len(bp)


def spath(f, ds):
    return CACHE / f"emb_{f}_strip_{ds}.npy"


def stripped_feat(f, ds):
    """Path of stripped embeddings if they exist, else the unstripped cache."""
    return spath(f, ds) if spath(f, ds).exists() else feat_path(f, ds)


def task(a):
    f, ds = a
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
    from reqcluster.ml import cluster, reduce

    X = l2(np.load(spath(f, ds)).astype(np.float32))
    y = load(ds)[1]
    n = len(y)
    cv = (StratifiedKFold if np.unique(y, return_counts=True)[1].min() >= 5 else KFold)(
        5, shuffle=True, random_state=0)
    p = cross_val_predict(LogisticRegression(C=10, max_iter=2000), X, y, cv=cv)
    mcs = cluster.auto_min_cluster_size(n)
    runs = [cluster.hdbscan(reduce.reduce(X, s)[0], mcs, 3)[0] for s in range(7)]
    aris = [ari(y, np.asarray(cluster.consensus(runs[s:s + 5], mcs, 3, X)[0] if 20 <= n <= 2000 else runs[s]))
            for s in range(3)]
    return {"feat": f, "ds": ds, "ceil_after": ari(y, p), "prod_after": float(np.mean(aris))}


def main():
    changed = {}
    texts = {}
    for ds in NAMES:
        t, y = load(ds)
        s, nb = strip(t)
        frac = float(np.mean([a != b for a, b in zip(t, s, strict=True)]))
        print(f"{ds:22s} n={len(t):5d} boilerplate_keys={nb:3d} rows_changed={frac:.3f}")
        if frac > 0:
            changed[ds] = s
            texts[ds] = s
    import torch
    from sentence_transformers import SentenceTransformer

    for f in FEATS:
        todo = [ds for ds in changed if not spath(f, ds).exists()]
        if not todo:
            continue
        mid, prompt, kw = MODELS[f]
        kw = {k: ({"torch_dtype": torch.float16} if k == "model_kwargs" else v) for k, v in kw.items()}
        m = SentenceTransformer(mid, device="cuda", trust_remote_code=True, **kw)
        for ds in todo:
            E = m.encode(changed[ds], batch_size=16, prompt=prompt, show_progress_bar=False)
            np.save(spath(f, ds), l2(np.asarray(E, dtype=np.float32)))
        del m
        gc.collect()
        torch.cuda.empty_cache()
    with get_context("spawn").Pool(8) as pool:
        res = pool.map(task, [(f, ds) for ds in changed for f in FEATS], chunksize=1)
    before = {(r["feat"], r["ds"]): r for r in map(json.loads, (CACHE / "ceiling_features.jsonl").read_text().splitlines())}
    rows = []
    for ds in NAMES:
        for f in FEATS:
            b = before[(f, ds)]
            r = next((x for x in res if x["feat"] == f and x["ds"] == ds), None)
            rows.append({"ds": ds, "feat": f, "changed": r is not None,
                         "prod_before": b["prod_ari"], "ceil_before": b["ceil_ari"],
                         "prod_after": r["prod_after"] if r else b["prod_ari"],
                         "ceil_after": r["ceil_after"] if r else b["ceil_ari"]})
    with open(CACHE / "boilerplate.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    for m in ("prod", "ceil"):
        t = df.pivot(index="ds", columns="feat", values=[f"{m}_before", f"{m}_after"])
        t = t.reindex(NAMES)
        print(m)
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()
