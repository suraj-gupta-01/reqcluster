import os, sys, time, json, warnings
S = "/tmp/claude-1000/-home-hikigaya-Desktop-reqcluster/0db9922a-4937-4abc-8178-f36d596d5423/scratchpad"
os.environ["NUMBA_CACHE_DIR"] = S + "/numba"
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score as ari
import umap

R = "/home/hikigaya/Desktop/reqcluster/data/"


def emb(name):
    p = f"{S}/emb_{name}.npy"
    d = pd.read_csv(R + name + "_requirements.csv", keep_default_na=False)
    y = d.module.astype(str).values
    if not os.path.exists(p):
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        np.save(p, m.encode(d.text.astype(str).tolist(), batch_size=64, normalize_embeddings=True))
    return np.load(p), y


umap.UMAP(n_components=2, n_epochs=10, compatibility_layout=True).fit_transform(np.random.rand(300, 20))
umap.UMAP(n_components=2, n_epochs=10, compatibility_layout=False).fit_transform(np.random.rand(300, 20))
VARIANTS = {
    "legacy_unseeded": dict(compatibility_layout=True),
    "legacy_seeded": dict(compatibility_layout=True, seed=True),
    "adam_seeded": dict(compatibility_layout=False, seed=True),
    "adam_seeded_spectral": dict(compatibility_layout=False, seed=True, init="spectral"),
}
only = os.environ.get("VARIANTS")
for name in sys.argv[1:]:
    X, y = emb(name)
    n = len(X)
    P = PCA(50, svd_solver="randomized", random_state=42).fit_transform(X)
    nn = min(80, max(15, n // 100))
    mcs = max(5, n // 50)
    base = dict(n_components=10, metric="cosine", n_neighbors=nn, min_dist=0.0, n_epochs=200, init="random")
    for v, kw in VARIANTS.items():
        if only and v not in only.split(","):
            continue
        aris, ts, labs = [], [], []
        opts = {**base, **{k: kv for k, kv in kw.items() if k != "seed"}}
        for s in (0, 1, 2):
            o = dict(opts)
            if kw.get("seed"):
                o["random_state"] = s
            t = time.time()
            Z = umap.UMAP(**o).fit_transform(P)
            ts.append(time.time() - t)
            L = HDBSCAN(min_cluster_size=mcs, min_samples=3, cluster_selection_method="eom").fit_predict(Z)
            aris.append(ari(y, L))
            labs.append(L)
        rep = None
        if kw.get("seed"):
            Z = umap.UMAP(**{**opts, "random_state": 0}).fit_transform(P)
            rep = bool((HDBSCAN(min_cluster_size=mcs, min_samples=3).fit_predict(Z) == labs[0]).all())
        print(json.dumps({"ds": name, "n": n, "v": v, "ari": round(float(np.mean(aris)), 3),
                          "std": round(float(np.std(aris)), 3), "t": round(float(np.mean(ts)), 1),
                          "repro": rep}), flush=True)
