"""6. Error behaviour."""
import pytest

from .conftest import topic_texts
from .helpers import assert_json_error

pytestmark = pytest.mark.acceptance

MISSING = 987654
RESOURCES = ["session", "clusters", "requirements", "graph", "metrics", "suggestions", "audit",
             "feedback_queue", "constraints", "feedback_export", "dependencies",
             "active_learning_queue", "quality_history", "enrich_status", "enrich_results"]


@pytest.mark.parametrize("name", RESOURCES)
def test_unknown_session_is_404(api, name):
    st, b = api.get(name, MISSING)
    assert_json_error(api, st, b, 404)


def test_unknown_session_cluster_and_export_404(api):
    run = api.cluster_start(MISSING)
    assert run.start_status == 404
    assert_json_error(api, 404, run.start_body)
    st, content, _ = api.export(MISSING, "csv")
    assert st == 404


@pytest.mark.parametrize("call", ["suggest", "dependencies_generate", "explain", "enrich"])
def test_unknown_provider_name_422(api, call):
    sid = api.upload_texts(topic_texts(2))
    fn = {
        "suggest": lambda: api.suggest(sid, provider_name="no-such-provider"),
        "dependencies_generate": lambda: api.dependencies_generate(sid, provider_name="no-such-provider"),
        "explain": lambda: api.explain(sid, 0, 1, provider_name="no-such-provider"),
        "enrich": lambda: api.enrich(sid, provider_name="no-such-provider"),
    }[call]
    st, b = fn()
    assert_json_error(api, st, b, 422)


def test_feedback_unknown_requirement_404(clustered, api):
    st, b = api.move(clustered, 99999999, -1)
    assert_json_error(api, st, b, 404)


def test_feedback_unknown_target_cluster_rejected(clustered, api):
    rid = api.all_requirements(clustered)[0]["id"]
    st, b = api.move(clustered, rid, 424242)
    assert st in (404, 422), (st, b)
    assert_json_error(api, st, b)


def test_review_unknown_feedback_404(clustered, api):
    st, b = api.review_feedback(clustered, 99999999, "approved")
    assert_json_error(api, st, b, 404)


def test_apply_unknown_suggestion_404(clustered, api):
    st, b = api.apply_suggestion(clustered, 99999999, "accept")
    assert_json_error(api, st, b, 404)


def test_explain_unknown_edge_404(dep_session, api):
    st, b = api.explain(dep_session, 5000, 5001)
    assert_json_error(api, st, b, 404)


def test_error_envelope_is_uniform(api):
    """Every error response parses as JSON with the same single envelope key."""
    sid = api.upload_texts(topic_texts(2))
    errs = [
        api.upload("x.txt", b"nope"),
        api.upload("bad.xlsx", b"junk"),
        api.get("clusters", MISSING),
        api.get("dependencies", MISSING),
        api.suggest(sid, provider_name="zzz"),
        api.move(sid, 99999999, -1),
        api.requirements(sid, sort_field="nope")[:2],
    ]
    shapes = set()
    for st, b in errs:
        assert 400 <= st < 500, (st, b)
        assert_json_error(api, st, b)
        shapes.add(tuple(sorted(b.keys())))
    assert len(shapes) == 1, shapes
