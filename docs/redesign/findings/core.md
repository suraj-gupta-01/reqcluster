HIGH embeddings.py:130 cache key "\n".join collision -> misaligned vectors; add model name/version [confirmed]
HIGH embeddings.py:36-70,175-208 silent random mock model on import fail, written to Redis/.npy caches
HIGH constrained_clustering.py:276-301 cannot-link repair breaks must-link; O(N*K) centroid per pair [confirmed]
HIGH split_suggest.py:~160-200 full silhouette per cluster O(N^2) -> sample_size=2000 seeded
HIGH graph.py:564,613-633 no-hnswlib fallback truncates to first 500 nodes, includes noise when n<=500 [confirmed] -> sklearn NearestNeighbors over all non-noise
MED reduction.py:48-56 random_state None for n>=4000 -> nondeterministic
MED reduction.py:60-66,76-90 PCA centring then cosine; GPU path ignores metric
MED split_suggest.py:~280 'and' should be 'or' (spread_threshold dead); GMM full cov overfit -> diag + BIC margin
MED domain_embeddings.py:270-301 Redis ignores use_cache/normalize; key lacks model
MED core/metrics.py:60-63,85,143 silhouette on 2D not 10D, unsampled; noise excluded from purity/hungarian
MED active_learning.py:388-390 unsampled silhouette
MED feedback_bridge.py:343-345 re-embeds base mode; zero-embedding fallback -> bogus constraints; dup req_id collapse
MED embeddings.py:157,223 mmap + non-atomic np.save race -> _atomic_save_npy
LOW embeddings.py:84-100,218 get_model no lock; generate_embeddings([]) crash; EMBEDDING_DIM hardcoded in cache
LOW artifact_extractor.py:228-257 _nlp flags set before load (race)
LOW artifact_extractor.py:700-712 P*C python loop -> np.argwhere
LOW dependency_tree.py:266-299 _break_cycles DFS per edge -> SCC
LOW labeling.py:120-123 list input TypeError; O(K*N) loop
LOW merge_suggest.py:~215 failed silhouette = 0.0 ranks as real
LOW clustering.py:8-29 no guard n<min_samples; min_cluster_size>n all noise [confirmed]
ARCH: fragmented non-atomic caches w/o model key; O(N^2) helpers scattered, env-dependent scale behaviour; noise semantics reimplemented x8, constraint repair contradictory
