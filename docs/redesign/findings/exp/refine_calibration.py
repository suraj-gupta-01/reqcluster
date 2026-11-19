"""R2-M6: coherence distribution (gold / merged pairs / random) and M5 suggestion precision.

Run: uv run python docs/redesign/findings/exp/refine_calibration.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score as ari

from reqcluster.ml import refine as R

ROOT = Path(__file__).resolve().parents[4]
CACHE = ROOT / "var" / "cache" / "gate"
NAMES = ["uav_fms_2k", "multidomain", "uav_fms_10k"]


def embeddings(name, texts):
    f = CACHE / f"emb_{name}.npy"
    if f.exists():
        return np.load(f)
    from sentence_transformers import SentenceTransformer
    E = SentenceTransformer("all-MiniLM-L6-v2", device="cpu").encode(
        texts, normalize_embeddings=True, show_progress_bar=False)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, E)
    return E


def stats(v):
    v = np.array(v)
    return f"n={len(v):3d} median={np.median(v):.3f} p10={np.percentile(v, 10):.3f} p90={np.percentile(v, 90):.3f}"


def precision(X, x10, gold, suggestions, apply):
    base, wins = ari(gold, gold), 0
    for s in suggestions:
        new = apply(gold.copy(), s)
        wins += ari(gold, new) > base
    return f"{wins}/{len(suggestions)}"


def old_suggestions(X, x10, gold):
    """Pre-M5 behaviour: no delta filtering (reimplemented from the unfiltered pieces)."""
    merges = [c for c in R.pairwise_similarity(R.unit_centroids(X, gold)) if c.centroid_similarity >= 0.75][:15]
    splits = []
    for cid in sorted(set(gold.tolist())):
        bm = R.bimodality(x10, gold, cid)
        if bm and bm.is_bimodal:
            splits.append((cid, bm.sub_labels))
    return merges[:5], splits[:5]


def apply_merge(g, c):
    g[g == c.cluster_b] = c.cluster_a
    return g


def apply_split(g, s):
    cid, sub = s
    idx = np.where(g == cid)[0]
    g[idx[np.asarray(sub) == 1]] = g.max() + 1
    return g


for name in NAMES:
    df = pd.read_csv(ROOT / "data" / f"{name}_requirements.csv")
    X = embeddings(name, df["text"].tolist())
    gold = pd.factorize(df["module"])[0].astype(np.int32)
    rng = np.random.default_rng(42)
    ids = sorted(set(gold.tolist()))
    sizes = {c: int((gold == c).sum()) for c in ids}
    g_coh = [R.coherence(X, gold, c, 42) for c in ids if sizes[c] >= 2]
    m_coh, r_coh = [], []
    for _ in range(100):
        a, b = rng.choice(ids, 2, replace=False)
        lab = np.where((gold == a) | (gold == b), 0, 1)
        m_coh.append(R.coherence(X, lab, 0, 42))
        rl = np.ones(len(gold), dtype=int)
        rl[rng.choice(len(gold), sizes[a] + sizes[b], replace=False)] = 0
        r_coh.append(R.coherence(X, rl, 0, 42))
    print(f"== {name} ({len(ids)} gold groups)")
    print("gold   ", stats(g_coh))
    print("merged ", stats(m_coh))
    print("random ", stats(r_coh))

    x10 = PCA(10, random_state=42).fit_transform(X).astype(np.float32)
    om, os_ = old_suggestions(X, x10, gold)
    nm = R.suggest_merges(X, x10, gold, seed=42)
    ns = R.suggest_splits(X, x10, gold, seed=42)
    print("merge precision before", precision(X, x10, gold, om, apply_merge),
          "after", precision(X, x10, gold, nm, apply_merge))
    print("split precision before", precision(X, x10, gold, os_, apply_split),
          "after", precision(X, x10, gold, [(s.cluster_id, s.sub_labels) for s in ns], apply_split))
