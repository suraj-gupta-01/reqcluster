"""Process-level native runtime setup (thread limits, warm-ups)."""
import logging
import sys
import threading

from reqcluster.config import Settings

log = logging.getLogger(__name__)


def configure_process(settings: Settings) -> None:
    # Do not point NUMBA_CACHE_DIR at {home}: an empty dir forces a full UMAP recompile (~90 s);
    # numba's default cache beside the installed packages persists (the Docker build warms it).
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


def warm_up_reducer(seed: int) -> threading.Thread:
    """The first UMAP call in a process spends ~20 s in numba JIT even with a warm cache (it does not
    avoid it). Pay it at startup on a daemon thread so the first clustering job does not."""
    def go():
        try:
            import numpy as np

            from reqcluster.ml import reduce
            reduce.reduce(np.random.default_rng(seed).standard_normal((40, 16)).astype(np.float32),
                          seed)
        except Exception:
            log.warning("reducer warm-up failed; the first clustering job will compile UMAP",
                        exc_info=True)

    t = threading.Thread(target=go, name="umap-warm-up", daemon=True)
    t.start()
    return t
