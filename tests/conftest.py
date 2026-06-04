import sys
import types
from importlib.util import find_spec
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

for path in (str(ROOT), str(BACKEND)):
    if path not in sys.path:
        sys.path.insert(0, path)


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
