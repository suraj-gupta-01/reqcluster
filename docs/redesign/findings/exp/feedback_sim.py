"""Feedback sim: ARI vs number of oracle expert corrections; current constraints mechanism vs propagation variants."""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari

from reqcluster.domain import ClusteringParams
from reqcluster.errors import Invalid
from reqcluster.ml import constraints, pipeline

ROOT = Path(__file__).resolve().parents[4]
GATE = ROOT / "var" / "cache" / "gate"
DS = ["aerospace", "bit_control", "sample", "uav_fms_phase2_audit", "dependency_gold", "adversarial", "multidomain", "promise"]
BUDGETS = [0, 10, 25, 50, 100]
SEEDS = [0, 1, 2]


def unit(M):
    return M / np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-12)


def targets(labels, gold):
    owner = {c: Counter(gold[labels == c]).most_common(1)[0][0] for c in set(labels) if c != -1}
    best = {}
    for c, g in owner.items():
        m = int((gold[labels == c] == g).sum())
        if g not in best or m > best[g][0]:
            best[g] = (m, c)
    return {g: v[1] for g, v in best.items()}


def expert(labels, probs, gold, B):
    tgt = targets(labels, gold)
    now = labels.copy()
    moves, checked = [], []
    for i in np.lexsort((np.arange(len(probs)), probs))[:B]:
        t = tgt.get(gold[i], -1)
        checked.append(int(i))
        if t != -1 and t != now[i]:
            moves.append((int(i), t, int(now[i])))
            now[i] = t
    return now, moves, checked


def mech_a(E, x10, base, moves):
    if not moves:
        return base.copy(), 0
    cur, must, cannot = base.copy(), [], []
    for i, new, old in moves:
        cur[i] = new
        m, c = constraints.from_correction(cur, i, new, old, X=x10)
        must += list(m)
        cannot += list(c)
    try:
        return np.asarray(constraints.apply(E, cur, must, cannot)[0]), 0
    except Invalid:
        return cur, 1


def mech_b(En, moved, checked, k=10, need=6, rounds=3):
    out = moved.copy()
    seed = np.zeros(len(out), bool)
    seed[checked] = True
    if not seed.any():
        return out
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    for _ in range(rounds):
        new = out.copy()
        for j in np.where(~seed & (out != -1))[0]:
            lab = Counter(int(out[n]) for n in nn[j] if seed[n] and out[n] != -1).most_common(1)
            if lab and lab[0][1] >= need:
                new[j] = lab[0][0]
        out = new
    return out


def mech_c(En, moved, checked, margin=0.02):
    out = moved.copy()
    seed = np.zeros(len(out), bool)
    seed[checked] = True
    if not seed.any():
        return out
    cents = {}
    for c in set(moved) - {-1}:
        m = (moved == c) & seed
        if not m.any():
            m = moved == c
        cents[c] = En[m].mean(0)
    ids = list(cents)
    C = unit(np.stack([cents[c] for c in ids]))
    sims = En @ C.T
    pos = {c: p for p, c in enumerate(ids)}
    for j in np.where(~seed & (moved != -1))[0]:
        b = int(sims[j].argmax())
        if sims[j, b] > sims[j, pos[moved[j]]] + margin:
            out[j] = ids[b]
    return out


def mech_d(E, En, moved, checked):
    out = moved.copy()
    if not checked:
        return out
    sim = En @ En[checked].T
    ck = np.asarray(checked)
    cents = {c: unit(En[moved == c].mean(0, keepdims=True))[0] for c in set(moved) - {-1}}
    seed = np.zeros(len(out), bool)
    seed[ck] = True
    for j in np.where(~seed & (moved != -1))[0]:
        b = int(sim[j].argmax())
        sj = ck[b]
        if moved[sj] != -1 and moved[sj] != moved[j] and sim[j, b] > En[j] @ cents[moved[j]]:
            out[j] = moved[sj]
    return out


def mech_e(En, moved, checked, k=10):
    out = moved.copy()
    if not checked:
        return out
    seed = np.zeros(len(out), bool)
    seed[checked] = True
    S = En @ En.T
    np.fill_diagonal(S, -np.inf)
    nn = np.argpartition(-S, k, axis=1)[:, :k]
    for j in np.where(~seed & (moved != -1))[0]:
        labs = [int(moved[n]) for n in nn[j] if seed[n]]
        if len(labs) >= 2:
            c, m = Counter(labs).most_common(1)[0]
            if c != -1 and c != moved[j] and m > len(labs) / 2:
                out[j] = c
    return out


def main():
    t0 = time.time()
    rows = []
    for ds in DS:
        df = pd.read_csv(ROOT / "data" / f"{ds}_requirements.csv", keep_default_na=False)
        gold = df["module"].to_numpy()
        E = np.load(sorted(GATE.glob(f"emb_minilm_prod_{ds}_*.npy"))[0])
        En = unit(E)
        raw = df["text"].astype(str).tolist()
        for s in SEEDS:
            res = pipeline.run(E, raw, ClusteringParams(), seed=s)
            lab = np.asarray(res.labels)
            for B in BUDGETS:
                moved, moves, checked = expert(lab, np.asarray(res.probabilities), gold, B)
                outs = {
                    "A": mech_a(E, res.x10, moved, moves),
                    "B": (mech_b(En, moved, checked), 0),
                    "C": (mech_c(En, moved, checked), 0),
                    "D": (mech_d(E, En, moved, checked), 0),
                    "E": (mech_e(En, moved, checked), 0),
                }
                tgt = targets(lab, gold)
                unchk = np.ones(len(lab), bool)
                unchk[checked] = False
                if B == 0:
                    assert all((o[0] == lab).all() for o in outs.values()), "B=0 must equal baseline"
                for m, (o, fail) in outs.items():
                    ch = unchk & (o != moved)
                    ok = sum(tgt.get(gold[j], -1) == o[j] for j in np.where(ch)[0])
                    rows.append(dict(ds=ds, seed=s, B=B, mech=m, ari=float(ari(gold, o)), moves=len(moves), fail=fail, nprop=int(ch.sum()), ncorrect=int(ok)))
        print(ds, "done", round(time.time() - t0), "s", flush=True)
    out = GATE / "feedback_sim.jsonl"
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    d = pd.DataFrame(rows)
    piv = d.pivot_table(index="ds", columns=["B", "mech"], values="ari").round(3)
    print(piv.to_string())
    print("overall ARI\n", d.pivot_table(index="mech", columns="B", values="ari").round(3).to_string())
    print("mean moves\n", d[d.mech == "A"].groupby("B").moves.mean().round(1).to_string())
    p = d[d.B.isin([25, 50])].groupby(["B", "mech"])[["nprop", "ncorrect"]].sum()
    p["precision"] = (p.ncorrect / p.nprop.where(p.nprop > 0)).round(3)
    print("propagation precision (sum over ds, seeds)\n", p.to_string())
    print("A apply failures:", int(d.fail.sum()), "runtime", round(time.time() - t0), "s")


if __name__ == "__main__":
    main()
