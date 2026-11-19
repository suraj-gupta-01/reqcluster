import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from .conftest import is_envelope, run_cluster, running_app, topic_texts, upload

pytestmark = pytest.mark.hardening
S = settings(max_examples=40, derandomize=True, deadline=None, suppress_health_check=list(HealthCheck))

BAD_IDS = ["-1", "0", "1", "2", "9223372036854775807", "abc", "1.5", "1e3", "%00", "%20", "٣", "NaN", "null", "true",
           "%E2%80%AE", "a" * 300, "../..", "%2e%2e", "😀"]
TEXTS = ["%", "_", "\\", "%_\\", "100%", "😀", "‮", "", " ", "a" * 5000, "'; DROP TABLE x;--", "\x00", "ß"]
QV = st.one_of(st.sampled_from(BAD_IDS + TEXTS + ["asc", "desc", "text", "req_id", "probability",
                                                  "cluster_id", "ordinal", "csv", "json", "pending",
                                                  "true", "false"]), st.text(max_size=20),
               st.integers(-10**15, 10**15).map(str))
JV = st.recursive(
    st.one_of(st.none(), st.booleans(), st.integers(-10**15, 10**15), st.floats(), st.text(max_size=20),
              st.sampled_from(TEXTS)),
    lambda ch: st.one_of(st.lists(ch, max_size=3), st.dictionaries(st.sampled_from(
        ["provider", "sim_threshold", "top_k", "source", "target", "force", "actor", "min_cluster_size",
         "min_samples", "embedding_mode", "seed", "requirement_id", "new_cluster_id", "comments",
         "applied_by", "top_n_merges", "extra", "x"]), ch, max_size=4)), max_leaves=8)
QNAMES = ["page", "page_size", "sort", "dir", "search", "cluster_id", "noise", "status", "format", "top_k",
          "x"]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    with running_app(tmp_path_factory.mktemp("api")) as (c, _):
        sid = upload(c, topic_texts(5))
        r = c.post(f"/api/sessions/{sid}/cluster", json={})
        assert r.status_code == 202
        job = r.json()["id"]
        yield c, sid, job


def pytest_generate_tests(metafunc):
    if "route" in metafunc.fixturenames:
        from reqcluster.app import create_app
        paths = create_app().openapi()["paths"]
        metafunc.parametrize("route", [(p, m.upper()) for p, ms in paths.items() for m in ms
                                       if m in ("get", "post", "put", "patch", "delete")],
                             ids=lambda r: f"{r[1]} {r[0]}")


@given(data=st.data())
@S
def test_fuzz_route(world, route, data):
    c, sid, job = world
    path, method = route
    url = path
    for name in [p for p in path.split("/") if p.startswith("{")]:
        n = name[1:-1]
        pool = BAD_IDS + ([] if method == "DELETE" and n == "sid" else [str(sid)])
        if n == "job_id":
            pool += [job]
        if n == "fmt":
            pool += ["csv", "pdf", "reqif", "sysml", "jama", "CSV", "csv/../x"]
        url = url.replace(name, data.draw(st.sampled_from(pool)).replace("/", "%2F"))
    params = {k: data.draw(QV) for k in data.draw(st.lists(st.sampled_from(QNAMES), max_size=4, unique=True))}
    kw = {}
    if method in ("POST", "PUT", "PATCH"):
        kind = data.draw(st.sampled_from(["json", "json", "raw", "none"]))
        if kind == "json":
            kw["json"] = data.draw(JV)
        elif kind == "raw":
            kw["content"] = data.draw(st.binary(max_size=200))
            kw["headers"] = {"content-type": data.draw(st.sampled_from(
                ["application/json", "text/plain", "multipart/form-data", "application/x-www-form-urlencoded"]))}
    r = c.request(method, url, params=params, **kw)
    assert r.status_code < 500, (method, url, params, kw, r.status_code, r.text[:200])
    if r.status_code >= 400:
        assert is_envelope(r), (method, url, r.status_code, r.text[:200])


def test_unknown_route_and_method(world):
    c, sid, _ = world
    for m, u in [("GET", "/api/nope"), ("PUT", f"/api/sessions/{sid}"), ("GET", "/"), ("POST", "/health")]:
        r = c.request(m, u)
        assert 400 <= r.status_code < 500 and is_envelope(r), (m, u, r.status_code)


def test_malformed_json_and_huge_body(world):
    c, sid, _ = world
    for body in (b"{", b"\xff\xfe", b"[" * 5000, b'{"a":' + b"1" * 100_000 + b"}"):
        r = c.post(f"/api/sessions/{sid}/dependencies", content=body,
                   headers={"content-type": "application/json"})
        assert 400 <= r.status_code < 500 and is_envelope(r), (body[:20], r.status_code)


HUGE = ["99999999999999999999", "9223372036854775808", "-9223372036854775809"]


@pytest.mark.xfail(strict=True, reason="FLAW: path/query/body ints are unbounded (api/*.py `sid: int`, "
                   "FeedbackIn.requirement_id/new_cluster_id, cluster_id query); ints > 2**63 reach "
                   "sqlite and raise OverflowError -> 500. Fix: Annotated[int, Path(ge=1, le=2**63-1)] "
                   "alias used everywhere, and bound the body/query ints")
def test_out_of_range_integers_are_client_errors(world):
    c, sid, job = world
    paths = c.app.openapi()["paths"]
    bad = []
    for path, ms in paths.items():
        for m in ms:
            for h in HUGE:
                url = path.replace("{sid}", h).replace("{cid}", h).replace("{fid}", h) \
                          .replace("{suggestion_id}", h).replace("{job_id}", job).replace("{fmt}", "csv")
                if "{" in url:
                    continue
                if h != HUGE[0] and path.count("{") > 1:
                    continue
                kw = {"json": {}} if m == "post" else {}
                r = c.request(m.upper(), url, **kw)
                if r.status_code >= 500:
                    bad.append((m.upper(), url))
    for body in ({"requirement_id": 10**30}, {"requirement_id": 1, "new_cluster_id": 10**30}):
        r = c.post(f"/api/sessions/{sid}/feedback", json=body)
        if r.status_code >= 500:
            bad.append(("POST", f"/feedback {body}"))
    for q in ({"cluster_id": 10**30}, {"page": 10**30}, {"page": 10**18, "page_size": 1000}):
        r = c.get(f"/api/sessions/{sid}/requirements", params=q)
        if r.status_code >= 500:
            bad.append(("GET", f"/requirements {q}"))
    assert not bad, (len(bad), bad)
