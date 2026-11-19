"""Process-level native runtime setup (threads, numba cache)."""
import os
import sys

from reqcluster.config import Settings


def configure_process(settings: Settings) -> None:
    cache = settings.cache_dir / "numba"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(cache))  # must precede umap/numba import
    apply_thread_limits(settings)


def apply_thread_limits(settings: Settings) -> None:
    n = settings.threads
    try:
        import numba

        numba.set_num_threads(min(n, numba.config.NUMBA_NUM_THREADS))
    except Exception:
        pass
    try:
        if "torch" in sys.modules:
            sys.modules["torch"].set_num_threads(n)
    except Exception:
        pass
    try:
        import threadpoolctl

        threadpoolctl.threadpool_limits(limits=n)
    except Exception:
        pass


def warm_up(settings: Settings) -> None:
    from reqcluster import embeddings

    embeddings.get_embedder().encode(["warm-up"], 1)
