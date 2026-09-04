import sys
import types
from importlib.util import find_spec
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(autouse=True)
def _isolate_embedding_cache(monkeypatch):
    """Keep unit tests hermetic from the live per-text Redis embedding cache.

    The cache (``core.embedding_cache``) holds a process-wide ``_redis_client``
    singleton.  If any test (or a prior live app run) connects it to a running
    Redis, later tests silently get cache *hits* instead of calling the model —
    which makes assertions like ``model.calls == 1`` flaky depending on whether
    the Docker Redis container happens to be up.  Force the cache into its
    documented no-op state for every test so the model path is always
    exercised; production is unaffected.
    """
    try:
        from core import embedding_cache
    except Exception:
        return
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setattr(embedding_cache, "_redis_client", None, raising=False)
    monkeypatch.setattr(embedding_cache, "_redis_unavailable", True, raising=False)


# torch and sentence_transformers are mocked *independently*.  Mocking them
# together (one try/except around both imports) meant a missing
# sentence_transformers would replace a perfectly good real torch with a stub —
# which then broke any other library that probes torch's real API surface
# (spaCy/thinc calls torch.cuda.device_count() at import time).
try:
    import torch  # noqa: F401
except (OSError, ImportError):
    import types

    # scipy's array-api-compat probes `torch.Tensor` unconditionally (via
    # getattr) whenever scipy is imported, so the attribute must exist even in
    # the mock or collection blows up with AttributeError.  thinc additionally
    # probes torch.cuda.device_count(), so provide a no-GPU stub for it.
    torch_mock = types.ModuleType("torch")
    torch_mock.Tensor = type("Tensor", (), {})
    cuda_mock = types.ModuleType("torch.cuda")
    cuda_mock.device_count = lambda: 0
    cuda_mock.is_available = lambda: False
    torch_mock.cuda = cuda_mock
    sys.modules["torch"] = torch_mock
    sys.modules["torch.cuda"] = cuda_mock

try:
    import sentence_transformers  # noqa: F401
except (OSError, ImportError):
    import types

    import numpy as np

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
