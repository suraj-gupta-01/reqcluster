import numpy as np

from reqcluster.store import EmbeddingStore, key


def test_key_depends_on_all_parts():
    assert key("m", "r", "t") != key("m", "r2", "t")
    assert key("m", "r", "t") == key("m", "r", "t")


def test_round_trip_and_chunking(tmp_path):
    s = EmbeddingStore(tmp_path / "sub" / "e.sqlite", 4)
    keys = [f"k{i}" for i in range(1000)]
    vecs = np.arange(4000, dtype=np.float32).reshape(1000, 4)
    s.put_many(keys, vecs)
    got = s.get_many(keys + ["missing"])
    assert len(got) == 1000
    assert np.array_equal(got["k999"], vecs[999])


def test_wrong_length_row_skipped(tmp_path):
    p = tmp_path / "e.sqlite"
    EmbeddingStore(p, 3).put_many(["a"], np.ones((1, 3), dtype=np.float32))
    assert EmbeddingStore(p, 4).get_many(["a"]) == {}


def test_corrupt_path_does_not_raise(tmp_path):
    p = tmp_path / "e.sqlite"
    p.write_bytes(b"not a database" * 100)
    s = EmbeddingStore(p, 2)
    assert s.get_many(["a"]) == {}
    s.put_many(["a"], np.ones((1, 2), dtype=np.float32))
    blocker = tmp_path / "file"
    blocker.write_text("x")
    bad = EmbeddingStore(blocker / "e.sqlite", 2)
    assert bad.get_many(["a"]) == {}
    bad.put_many(["a"], np.ones((1, 2), dtype=np.float32))
