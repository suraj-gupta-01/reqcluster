import numpy as np

from reqcluster.ml import reduce


def test_exact_knn_path_emits_no_umap_warning():
    # we never call UMAP.transform, so its "no search index" notice was noise on every 1k+ upload
    import warnings

    import umap  # noqa: F401  its own import warns (no TensorFlow); not what this test checks

    X = np.random.default_rng(0).standard_normal((reduce.EXACT_KNN_MIN, 16)).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reduce.reduce(X, 0)
