"""Intrinsic (always) and extrinsic (with ground truth) clustering metrics. Noise = -1."""
from collections import Counter

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_rand_score,
    homogeneity_completeness_v_measure,
    normalized_mutual_info_score,
)

from . import labels as L
from . import sampling


def has_ground_truth(gt: list[str], n: int) -> bool:
    """'General'/blank labels do not count; need >= 2 distinct real groups."""
    if len(gt) != n:
        return False
    return len({g.strip().lower() for g in gt if g.strip() and g.strip().lower() != "general"}) >= 2


def _hungarian_matched(cl: np.ndarray, tr: np.ndarray) -> int:
    """Points matched by the optimal 1-to-1 cluster->group assignment (0 if < 2 clusters)."""
    cids, gids = sorted(set(cl.tolist())), sorted(set(tr.tolist()))
    if len(cids) < 2:
        return 0
    ci, gi = {c: i for i, c in enumerate(cids)}, {g: j for j, g in enumerate(gids)}
    cost = np.zeros((len(cids), len(gids)), dtype=np.int64)
    for c, t in zip(cl.tolist(), tr.tolist(), strict=True):
        cost[ci[c], gi[t]] += 1
    r, c = linear_sum_assignment(-cost)
    return int(cost[r, c].sum())


def _confusion(cl: np.ndarray, tr: np.ndarray) -> dict:
    cids, gnames = sorted(set(cl.tolist())), sorted(set(tr.tolist()))
    ci, gi = {c: i for i, c in enumerate(cids)}, {g: j for j, g in enumerate(gnames)}
    m = [[0] * len(gnames) for _ in cids]
    for c, t in zip(cl.tolist(), tr.tolist(), strict=True):
        m[ci[c]][gi[t]] += 1
    return {"rows": m, "cluster_ids": cids, "group_names": gnames}


def compute(labels: np.ndarray, x10: np.ndarray | None, seed: int,
            ground_truth: list[str] | None = None) -> dict:
    labels = L.normalize(labels)
    n = len(labels)
    noise = int((labels == -1).sum())
    rate = noise / n if n else 0.0
    out = {
        "n": n,
        "n_clusters": len(L.cluster_ids(labels)),
        "noise_count": noise,
        "noise_rate": rate,
        "coverage": 1 - rate,
        "silhouette": None if x10 is None else sampling.silhouette(x10, labels, seed=seed),
        "has_ground_truth": False,
    }
    gt = [str(g or "").strip() for g in (ground_truth or [])]
    mask = L.clustered_mask(labels)
    if not has_ground_truth(gt, n) or mask.sum() < 2:
        return out

    cl, tr = labels[mask], np.asarray(gt)[mask]
    hom, comp, v = homogeneity_completeness_v_measure(tr, cl)
    by_cluster: dict[int, Counter] = {}
    for c, t in zip(cl.tolist(), tr.tolist(), strict=True):
        by_cluster.setdefault(c, Counter())[t] += 1
    purity = sum(c.most_common(1)[0][1] for c in by_cluster.values()) / len(cl)
    matched = _hungarian_matched(cl, tr)
    out.update({
        "has_ground_truth": True,
        "n_true_groups": len(set(tr.tolist())),
        "ari": float(adjusted_rand_score(tr, cl)),
        "nmi": float(normalized_mutual_info_score(tr, cl)),
        "homogeneity": float(hom),
        "completeness": float(comp),
        "v_measure": float(v),
        "purity": purity,
        "hungarian_accuracy": matched / len(cl),
        "accuracy_noise_inclusive": matched / n,  # noise counts as wrong
        "confusion": _confusion(cl, tr),
    })
    return out
