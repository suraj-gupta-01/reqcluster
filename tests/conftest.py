import sys
import types
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)


try:
    import torch
    import sentence_transformers
except (OSError, ImportError) as e:
    import types
    import numpy as np

    # Mock torch
    torch_mock = types.ModuleType("torch")
    sys.modules["torch"] = torch_mock

    # Mock sentence_transformers
    st_mock = types.ModuleType("sentence_transformers")
    class DummySentenceTransformer:
        def __init__(self, *args, **kwargs):
            self.model_name = args[0] if args else "mock-model"
        def encode(self, batch, **kwargs):
            # Return dummy 384-dimensional embeddings
            n = len(batch) if isinstance(batch, list) else 1
            vectors = np.zeros((n, 384), dtype=np.float32)
            if kwargs.get("normalize_embeddings", False):
                # Put a dummy 1.0 in the first dimension so norm is 1
                vectors[:, 0] = 1.0
            return vectors

    st_mock.SentenceTransformer = DummySentenceTransformer
    sys.modules["sentence_transformers"] = st_mock


if find_spec("hdbscan") is None:
    hdbscan = types.ModuleType("hdbscan")

    class HDBSCAN:
        def __init__(self, *args, **kwargs):
            raise ImportError("hdbscan is required for real clustering runs.")

    hdbscan.HDBSCAN = HDBSCAN
    sys.modules["hdbscan"] = hdbscan


if find_spec("umap") is None:
    umap = types.ModuleType("umap")

    class UMAP:
        def __init__(self, *args, **kwargs):
            raise ImportError("umap-learn is required for real reduction runs.")

    umap.UMAP = UMAP
    sys.modules["umap"] = umap
