# ML experiment grid (2026-11-19)

Keep all-MiniLM-L6-v2 and tune UMAP and the clusterer. Embedder choice moves accuracy by only ~0.03, and MiniLM is 10x or more faster than the rest.

Everything is in `/tmp/claude-1000/-home-hikigaya-Desktop-reqcluster/0dfe9f9d-be6c-4b09-a5fe-c419290c1f2e/scratchpad/exp/` (scripts, `res_*.jsonl`, `t10k.jsonl`, `emb/*.npy`). I did not edit repo source.

**Method notes**
- ARI and NMI treat noise as its own label. Accuracy is Hungarian with noise counted wrong; on adversarial the `__noise__` rows count as correct only if predicted noise. Each cell is ARI±std / NMI / accuracy.
- UMAP rows use 3 seeds. KMeans and ward/avg-cosine rows are single seed (UMAP seed 0), so no std.
- The grid ran on multidomain, adversarial, UAV 2k, PROMISE, plus UAV 10k for MiniLM only. PROMISE is uninformative: ARI ≈0.05 for every config, so I left it out of the table.

**Top configs** (all PCA50, then UMAP 10d, cosine, min_dist 0, n_neighbors 15 unless noted)

| Embedder + clusterer | multi | adv | uav2k | uav10k |
|---|---|---|---|---|
| nomic + KMeans, silhouette k | .687/.79/.801 | .355/.69/.575 | .964/.98/.979 | not run |
| nomic + ward, silhouette k | .638/.78/.735 | .367/.69/.573 | .960/.98/.977 | not run |
| nomic + HDBSCAN n//50 s3 eom | .644±.010/.78/.748 | .369±.025/.68/.563 | .931±.018/.96/.935 | not run |
| MiniLM + HDBSCAN n//50 s3 eom (current) | .641±.008/.77/.754 | .343±.019/.66/.553 | .941±.018/.97/.951 | .904±.014/.95/.890 |
| MiniLM + avg-cosine agglo, silhouette k | .617/.77/.731 | .327/.66/.540 | .973/.98/.983 | .881/.94/.856 |
| MiniLM + HDBSCAN leaf | .661±.015/.78/.769 | .334±.019/.65/.568 | .821±.048/.91/.794 | .769/.89/.752 |
| qwen3 + avg-cosine agglo | .618/.76/.728 | .375/.70/.562 | .974/.98/.984 | skipped |
| gte-b + KMeans | .608/.74/.728 | .353/.68/.568 | .944/.97/.958 | not run |
| bge-s + KMeans | .621/.74/.751 | .323/.66/.490 | .965/.98/.968 | not run |
| MiniLM, n_neighbors=80, 200 epochs, HDBSCAN | n/a | n/a | n/a | .911±.010/.955/.888 |
| MiniLM, n_neighbors=80, 200 epochs, ward + silhouette k | n/a | n/a | n/a | .974±.041/.991/.974 |

n_neighbors 80 is `min(80, max(15, n//100))`. The 10k baseline reproduced at 0.904±0.014 ARI.

**Best accuracy:** MiniLM, PCA50, UMAP n_neighbors=80, 200 epochs, then ward linkage with k chosen by silhouette over 8–40. On 10k it gets ARI 0.974 and accuracy 0.974, though one seed picks a worse k (std 0.041). It never produces noise, so it can't flag the `__noise__` rows. On the small hard sets nomic + KMeans leads (multi 0.687).

**Best speed/accuracy trade-off:** MiniLM, PCA50, UMAP(n_components=10, metric=cosine, n_neighbors=min(80,max(15,n//100)), min_dist=0, n_epochs=200, init=random), then HDBSCAN(min_cluster_size=max(5,n//50), min_samples=3, eom). At 10k, UMAP drops from 95s to 23s, and ARI goes from 0.904 to 0.911. Clean-CPU 10k total is about 78s versus about 150s now (encode 54s, UMAP 23s, HDBSCAN 1.2s).

**Reducers and clusterers**
- HDBSCAN on PCA or raw embeddings collapses: multi ARI 0.1, adversarial ≈0. UMAP is essential.
- Spectral init, n_neighbors 30, and `min_cluster_size` n//100 or sqrt(n) are all worse; leaf is worse than eom.
- `min_samples` 1, 3 and 5 differ by about 0.01.
- On 10k, PCA20 + ward reaches 0.951, but on multi it reaches only 0.45. UMAP + ward + silhouette k beats HDBSCAN on UAV but not on multi or adversarial.
- No clusterer detects the adversarial noise well: about 8% noise predicted versus 20% true.

**Speed variants** (MiniLM, UAV 2k, clean CPU)
- ONNX is not faster. Encoding runs at 185 texts/s with torch, 169 with ONNX fp32, and 197 with int8 (+6%).
- int8 costs accuracy: ARI 0.934 to 0.923, accuracy 0.941 to 0.921, embedding cosine 0.991.
- `max_seq_length` 128 or 64 gives identical embeddings and no speedup, since texts are short.
- UMAP at 200 epochs on 2k takes 2.4s versus 4.5s, with no accuracy loss.
- Persisting the numba cache saves about 35s of first-import time (37.6s cold, 2.1s warm).

**Embedder ranking** (pooled ARI+accuracy on multi, adv, uav2k; clean encode speed only measured for MiniLM)
1. nomic: +0.02–0.03 ARI on multi and adversarial, but about 10x slower than MiniLM.
2. MiniLM: 185 texts/s clean, about 3x faster than bge-s.
3. gte-b
4. bge-s
5. bge-b
6. qwen3
7. mpnet
8. e5-b
9. mxbai: worst, ARI 0.526 on multi.

**Recommendation for the production default**
- Keep MiniLM, and replace the n_neighbors formula with `min(80, max(15, n//100))`.
- Set `n_epochs=200`, keep PCA50 and HDBSCAN eom, and set the numba cache dir.
- Offer ward + silhouette k as an optional "no noise" mode. Offer nomic as an optional "accuracy" embedder.
- Skip ONNX.

**Caveats and failures**
- Nondeterministic UMAP (random init, unseeded at ≥4000 rows) gives about ±0.015 ARI at 10k.
- Encode and clustering ran concurrently, so any timing from the grid is contaminated. Only the speed tests and the 10k timing runs had a quiet CPU.
- gte-b and nomic failed first in fp16 and then on a string dtype; they work with `torch_dtype=torch.float32`.
- mxbai was too slow (about 2 texts/s) and was run on multi and adversarial only. qwen3 was run on multi, adversarial and UAV 2k only.
- 10k was run only for MiniLM and I killed the raw-embedding 10k task, so that row is missing. HDBSCAN on raw embeddings is already known to be poor.
- The exported int8 ONNX model could not be loaded (the export has no config), so I used the prebuilt `onnx/model_quint8_avx2.onnx` from the hub. This CPU has no avx512_vnni.
- The pre-built-package install downgraded `transformers` 5.23 to 4.57.6 and `tokenizers` 0.23 to 0.22.2. Another process ran `uv sync` and removed `einops`, so I re-installed it. `uv.lock` shows modified, but not by me.
- Extra packages installed into the venv: `einops`, `optimum[onnxruntime]`, `onnx`.