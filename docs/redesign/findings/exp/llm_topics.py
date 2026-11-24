"""LLM topic discovery + classification (Ollama) vs gold modules; also embedding-nearest-topic."""
import os
import gc
import json
import random
import subprocess
import time
import urllib.request
from collections import Counter

import numpy as np
import pandas as pd
from boilerplate import stripped_feat, strip
from ceiling_features import CACHE, MODELS, ari, l2, load

URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen3.5:4b"
DSS = os.environ.get("DS", "aerospace,bit_control,sample,uav_fms_phase2_audit,adversarial,uav_fms_stress_600").split(",")
SEEDS = tuple(int(x) for x in os.environ.get("SEEDS", "0").split(","))  # ~2 s/row per seed
DIR = CACHE / "llm_topics"


def chat(prompt, seed):
    body = {"model": MODEL, "think": False, "format": "json", "stream": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0.2, "seed": seed, "num_ctx": 16384}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"]


def cached(ds, S, step, calls):
    """calls: list of (tag, prompt). Cache raw responses per tag; returns (dict tag->raw, secs)."""
    p = DIR / f"{ds}_{S}_{step}.json"
    c = json.loads(p.read_text()) if p.exists() else {"raw": {}, "secs": 0.0}

    def get(tag, prompt):
        if tag not in c["raw"]:
            t = time.time()
            try:
                c["raw"][tag] = chat(prompt, S)
            except Exception as e:  # noqa: BLE001
                c["raw"][tag] = f"ERROR {e!r}"
            c["secs"] += time.time() - t
            p.write_text(json.dumps(c))
        return c["raw"][tag]
    return get, c


def parse(raw, key):
    try:
        v = json.loads(raw)[key]
        return v if isinstance(v, list) else []
    except Exception:  # noqa: BLE001
        return []


def taxonomy(ds, S, texts):
    idx = list(range(len(texts))) if len(texts) <= 200 else sorted(random.Random(S).sample(range(len(texts)), 200))
    lst = "\n".join(f"{j + 1}. {texts[i]}" for j, i in enumerate(idx))
    prompt = ("Below are numbered system requirements.\n" + lst +
              "\n\nIdentify 4 to 20 non-overlapping subsystems or functional areas that partition these "
              "requirements, so that every requirement fits exactly one. Reply with JSON only: "
              '{"topics":[{"name": "...", "description": "one sentence"}]}')
    get, c = cached(ds, S, "tax", None)
    raw = get("tax", prompt)
    if not [t for t in parse(raw, "topics") if isinstance(t, dict) and t.get("name")]:
        raw = get("tax_retry", prompt)
    tops = [t for t in parse(raw, "topics") if isinstance(t, dict) and t.get("name")]
    return [(str(t["name"]), str(t.get("description", ""))) for t in tops], c["secs"]


def classify(ds, S, texts, topics):
    tl = "\n".join(f"{j + 1}. {n}: {d}" for j, (n, d) in enumerate(topics))
    get, c = cached(ds, S, "cls", None)
    lab = np.full(len(texts), -1)
    for b in range(0, len(texts), 20):
        ids = range(b, min(b + 20, len(texts)))
        rq = "\n".join(f"{i + 1}. {texts[i]}" for i in ids)
        prompt = (f"Topics:\n{tl}\n\nRequirements:\n{rq}\n\nAssign each requirement to exactly one topic. "
                  'Reply with JSON only: {"assignments":[{"id": <requirement number>, "topic": <topic number>}]}')
        for attempt in range(2):
            raw = get(f"b{b}_a{attempt}", prompt)
            for a in parse(raw, "assignments"):
                try:
                    i, t = int(a["id"]) - 1, int(a["topic"]) - 1
                except Exception:  # noqa: BLE001
                    continue
                if b <= i < b + 20 and i < len(texts) and 0 <= t < len(topics):
                    lab[i] = t
            if (lab[b:b + 20] >= 0).all():
                break
    return lab, c["secs"]


def ensure_ollama():
    try:
        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
    except Exception:  # noqa: BLE001
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(8)


def unload():
    body = {"model": MODEL, "keep_alive": 0, "messages": []}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=60).read()
    time.sleep(3)


def prod_labels(ds):
    from reqcluster.ml import cluster, reduce
    X = l2(np.load(stripped_feat("qwen3i", ds)).astype(np.float32))
    n = len(X)
    mcs = cluster.auto_min_cluster_size(n)
    runs = [cluster.hdbscan(reduce.reduce(X, s)[0], mcs, 3)[0] for s in range(5)]
    return np.asarray(cluster.consensus(runs, mcs, 3, X)[0] if 20 <= n <= 2000 else runs[0])


def x_prod(lab, prod):
    out = lab.copy()
    for c in set(prod) - {-1}:
        m = prod == c
        v = [x for x in lab[m] if x >= 0]
        if v:
            out[m & (lab == -1)] = Counter(v).most_common(1)[0][0]
    return out


def main():
    import torch
    from sentence_transformers import SentenceTransformer
    ensure_ollama()
    DIR.mkdir(parents=True, exist_ok=True)
    state = {}
    seeds = SEEDS
    for ds in DSS:
        texts, y = load(ds)
        texts = strip(texts)[0]
        for S in seeds:
            topics, s1 = taxonomy(ds, S, texts)
            if not topics:
                print(f"FAIL taxonomy {ds} S={S}")
                continue
            lab, s2 = classify(ds, S, texts, topics)
            state[(ds, S)] = (topics, lab, s1 + s2)
            print(f"{ds} S={S} topics={len(topics)} invalid={(lab == -1).mean():.3f} secs={s1 + s2:.0f}", flush=True)
    unload()
    prod = {ds: prod_labels(ds) for ds in DSS}
    rows = []
    embs = {}
    for f in ("qwen3i", "minilm"):
        mid, prompt, kw = MODELS[f]
        kw = {k: ({"torch_dtype": torch.float16} if k == "model_kwargs" else v) for k, v in kw.items()}
        m = SentenceTransformer(mid, device="cuda", trust_remote_code=True, **kw)
        for (ds, S), (topics, _, _) in state.items():
            T = l2(np.asarray(m.encode([f"{n}: {d}" for n, d in topics], show_progress_bar=False), dtype=np.float32))
            embs[(f, ds, S)] = T
        del m
        gc.collect()
        torch.cuda.empty_cache()
    for (ds, S), (topics, lab, secs) in state.items():
        y = load(ds)[1]
        k = len(set(y))
        v = {"llm_cls": lab, "llm_cls_x_prod": x_prod(lab, prod[ds])}
        for f in ("qwen3i", "minilm"):
            X = l2(np.load(stripped_feat(f, ds)).astype(np.float32))
            v[f"llm_emb_{f}"] = (X @ embs[(f, ds, S)].T).argmax(1)
        for name, p in v.items():
            rows.append({"ds": ds, "S": S, "variant": name, "ari": ari(y, p), "n_topics": len(topics),
                         "k": k, "invalid": float((lab == -1).mean()), "secs": secs})
    with open(CACHE / "llm_topics.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    df = pd.DataFrame(rows)
    g = df.groupby(["ds", "variant"]).agg(mean_ari=("ari", "mean"), min_ari=("ari", "min"),
                                          topics=("n_topics", "mean"), k=("k", "first"),
                                          invalid=("invalid", "mean"), secs=("secs", "sum")).round(3)
    print(g.to_string())


if __name__ == "__main__":
    main()
