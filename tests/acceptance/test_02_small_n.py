"""2. Small-N and degenerate clustering."""
import pytest

from .conftest import topic_texts
from .helpers import assert_coherent

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("n", [1, 2, 4, 5])
def test_small_n_completes(api, n):
    texts = topic_texts(per_topic=2)[::2][:n] if n <= 4 else topic_texts(per_topic=2)[:n]
    sid = api.upload_texts(texts)
    res = api.cluster(sid)
    assert res["http"] in (200, 202), res
    assert res["state"] == "done", res
    assert_coherent(api, sid, expect_n=n)


def test_identical_texts_collapse_and_complete(api):
    sid = api.upload_texts(["The brake shall engage within 5 ms"] * 10)
    res = api.cluster(sid)
    assert res["state"] == "done", res
    assert_coherent(api, sid, expect_n=1)


def test_near_identical_texts_complete(api):
    sid = api.upload_texts([f"The brake shall engage within 5 ms case {i}" for i in range(12)])
    res = api.cluster(sid)
    assert res["state"] == "done", res
    assert_coherent(api, sid, expect_n=12)


def test_all_noise_reported_coherently(api):
    sid = api.upload_texts(topic_texts(per_topic=3))
    res = api.cluster(sid, min_cluster_size=60)  # larger than N -> all noise (or one cluster)
    assert res["state"] == "done", res
    reqs, clusters = assert_coherent(api, sid, expect_n=12)
    if not clusters:
        assert all(r["is_noise"] for r in reqs)
        st, s = api.session(sid)
        assert s["noise_count"] == 12 and s["total_clusters"] == 0


def test_min_cluster_size_larger_than_n_does_not_500(api):
    sid = api.upload_texts(topic_texts(per_topic=2)[:6])
    res = api.cluster(sid, min_cluster_size=500)
    assert res["state"] == "done", res
    assert_coherent(api, sid, expect_n=6)
