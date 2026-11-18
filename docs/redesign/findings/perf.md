12 cores CPU torch. 10k cold: embed 72.8s, UMAP 45-56s, HDBSCAN 0.9, graph 1.4, labels .07; total 131s cold / 48s warm.
2k: embed 18.1, UMAP 27-29 (JIT fixed cost), total 47 cold/27 warm. dep tree 2k 0.58s warm (4.5 first: model load via resolve_near_matches).
Accuracy (module label, 17 cls): 2k ARI .942 NMI .969; 10k ARI .886-.901 (nondeterministic, random_state None n>=4000).
spaCy en_core_web_sm NOT installed -> regex fallback.
Speedups: (1) PCA/SVD 10d instead of UMAP for clustering, UMAP only 2D viz (test ARI); (2) UMAP epochs/NUMBA_CACHE_DIR/seed; (3) embedding: GPU / ONNX int8 / max_seq_length / length-sorted batches; (4) load model once at startup; (5) Rust/numba NOT worth (stages <2s); (6) hnswlib/numba thread oversubscription spike.
