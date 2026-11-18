"""8. Concurrency."""
import time

import pytest

from .conftest import topic_texts
from .helpers import assert_coherent

pytestmark = pytest.mark.acceptance


def test_two_concurrent_cluster_requests(api, embedder):
    sid = api.upload_texts(topic_texts(6))
    embedder.hold()
    a = api.cluster_start(sid, min_cluster_size=3)
    try:
        assert embedder.entered.wait(30)
        b = api.cluster_start(sid, min_cluster_size=3)
        time.sleep(0.5)
    finally:
        embedder.release()
    ra, rb = api.wait(a), api.wait(b)
    assert ra["http"] in (200, 202), ra
    assert rb["http"] in (200, 202, 409), rb
    assert ra["state"] == "done"
    assert rb["state"] in ("done", "rejected")
    assert_coherent(api, sid, expect_n=24)


def test_enrichment_while_clustering_is_409(api, embedder):
    sid = api.upload_texts(topic_texts(6))
    embedder.hold()
    run = api.cluster_start(sid, min_cluster_size=3)
    try:
        assert embedder.entered.wait(30)
        st, b = api.enrich(sid, provider_name="mock")
        assert st == 409, (st, b)
    finally:
        embedder.release()
    assert api.wait(run)["state"] == "done"


def test_mutations_during_clustering_not_500(api, embedder):
    sid = api.upload_texts(topic_texts(6))
    embedder.hold()
    run = api.cluster_start(sid, min_cluster_size=3)
    try:
        assert embedder.entered.wait(30)
        rid = api.all_requirements(sid)[0]["id"]
        st, _ = api.move(sid, rid, -1)
        assert st < 500
        assert api.suggest(sid)[0] < 500
        assert api.dependencies_generate(sid)[0] < 500
    finally:
        embedder.release()
    assert api.wait(run)["state"] in ("done", "error")
