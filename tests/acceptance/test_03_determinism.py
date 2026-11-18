"""3. Determinism."""
import pytest

from .conftest import topic_texts

pytestmark = pytest.mark.acceptance


def test_same_upload_same_config_same_labels(api):
    texts = topic_texts(per_topic=8)
    a = api.upload_texts(texts)
    b = api.upload_texts(texts)
    assert api.cluster(a)["state"] == "done"
    assert api.cluster(b)["state"] == "done"
    assert api.labels(a) == api.labels(b)


def test_recluster_same_session_same_config_same_labels(api):
    sid = api.upload_texts(topic_texts(per_topic=8))
    assert api.cluster(sid, min_cluster_size=4)["state"] == "done"
    first = api.labels(sid)
    assert api.cluster(sid, min_cluster_size=4)["state"] == "done"
    assert api.labels(sid) == first
    assert len(set(first) - {-1}) >= 1
