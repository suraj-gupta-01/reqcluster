import sys
import types

import reqcluster

from reqcluster import runtime
from reqcluster.config import get_settings


def test_apply_thread_limits_numba(monkeypatch):
    calls = []
    fake = types.ModuleType("numba")
    fake.set_num_threads = calls.append
    fake.config = types.SimpleNamespace(NUMBA_NUM_THREADS=64)
    monkeypatch.setitem(sys.modules, "numba", fake)
    s = get_settings()
    runtime.apply_thread_limits(s)
    assert calls == [min(s.threads, 64)]


def test_apply_thread_limits_missing_numba(monkeypatch):
    monkeypatch.setitem(sys.modules, "numba", None)
    runtime.apply_thread_limits(get_settings())


def test_warm_up(monkeypatch):
    seen = []
    fake = types.ModuleType("reqcluster.embeddings")
    fake.get_embedder = lambda: types.SimpleNamespace(
        encode=lambda t, b: seen.append((t, b))
    )
    monkeypatch.setitem(sys.modules, "reqcluster.embeddings", fake)
    monkeypatch.setattr(reqcluster, "embeddings", fake, raising=False)
    runtime.warm_up(get_settings())
    assert seen == [(["warm-up"], 1)]
