"""Benchmark SBERT embedding throughput on the detected accelerator.

Auto-detects the device (via torch.cuda), loads the model on it, warms
it up, then times encoding N texts and reports throughput. Force a device with
--device to compare CPU vs GPU on the same machine.

Standalone: only needs `sentence-transformers`, `torch`, `numpy` - so it runs in
the isolated GPU venv too.

Examples:
  python scripts/benchmark_embeddings.py --n 5000
  python scripts/benchmark_embeddings.py --n 5000 --device cuda --repeats 3
  python scripts/benchmark_embeddings.py --n 5000 --device cpu
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

from reqcluster.embeddings import adaptive_batch_size


def make_texts(n: int) -> list[str]:
    subs = ["thermal unit", "power unit", "transceiver", "flight computer", "star tracker",
            "valve", "fault manager", "solar drive", "cooling loop", "battery manager"]
    verbs = ["activate", "regulate", "report", "measure", "limit", "record", "transmit", "verify"]
    objs = ["temperature", "voltage", "pressure", "angular rate", "state of charge", "fault flag",
            "telemetry frame", "attitude", "duty cycle", "data rate"]
    return [f"The {subs[i % len(subs)]} shall {verbs[i % len(verbs)]} the {objs[i % len(objs)]} "
            f"within {20 + (i % 480)} units during phase {i % 7}." for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--model", default="all-MiniLM-L6-v2")
    ap.add_argument("--batch", type=int, default=0, help="0 = auto from device")
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    import torch

    cuda = torch.cuda.is_available()
    chosen = ("cuda:0" if cuda else "cpu") if args.device == "auto" else (
        "cuda:0" if args.device == "cuda" else "cpu")
    batch = args.batch or adaptive_batch_size(args.n)

    if args.device == "cuda" and not cuda:
        print("[error] --device cuda requested but CUDA is unavailable.")
        sys.exit(2)

    print(f"running on: {chosen} | model={args.model} | n={args.n} | batch={batch} | repeats={args.repeats}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model, device=chosen)

    texts = make_texts(args.n)
    # warm up (model load + JIT/CUDA context) - excluded from timing
    model.encode(texts[:64], batch_size=batch, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)

    times = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        emb = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=False)
        times.append(time.perf_counter() - t0)

    med = statistics.median(times)
    print(f"RESULT device={chosen} n={args.n} "
          f"median={med:.2f}s min={min(times):.2f}s max={max(times):.2f}s "
          f"throughput={args.n / med:,.0f} req/s ms_per_req={1000 * med / args.n:.3f} "
          f"shape={emb.shape}")


if __name__ == "__main__":
    main()
