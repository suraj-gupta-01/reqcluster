"""Black-box acceptance fixtures.

ONE swap point: the `api` fixture. Tests only call `api.*` methods and the
`embedder` / `fake_llm` fixtures; never URLs, error shapes or internals.
A future `NewApi` must implement the same methods, then `api` returns it.
Adapter conventions
  * methods return (http_status, body) unless noted; body = parsed JSON or raw text
  * cluster()/enrich() block until the work is finished (job polling hidden)
  * error_message(body) -> str|None, normalising the error envelope
"""
import hashlib
import io
import json
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ---------------------------------------------------------------- test data

TOPICS = [
    ("brake", "pedal", "caliper", "hydraulic"),
    ("battery", "cell", "charger", "voltage"),
    ("display", "screen", "pixel", "backlight"),
    ("radio", "antenna", "signal", "frequency"),
]


def topic_texts(per_topic=10, topics=TOPICS):
    out = []
    for t in topics:
        for i in range(per_topic):
            out.append(
                f"The {t[0]} {t[1]} {t[2]} {t[3]} subsystem shall report {t[0]} {t[3]} status "
                f"code {i} within {10 + i} milliseconds"
            )
    return out


DEP_TEXTS = [
    "The thermal sensor shall generate a temperature reading.",
    "The cooling controller shall use the temperature reading as defined in REQ-001.",
    "The fan shall activate once the temperature reading exceeds the limit.",
    "The power module shall provide a regulated voltage output.",
    "The processor shall require the regulated voltage to operate.",
    "The system shall log voltage faults as specified in REQ-004.",
]


def make_csv(texts, ids=True, header="text", extra=None, newline="\n"):
    """Build CSV bytes (utf-8) with proper quoting."""
    import csv

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator=newline)
    cols = (["id"] if ids else []) + [header] + list((extra or {}).keys())
    w.writerow(cols)
    for i, t in enumerate(texts):
        row = ([f"REQ-{i + 1:03d}"] if ids else []) + [t] + [v[i] for v in (extra or {}).values()]
        w.writerow(row)
    return buf.getvalue().encode("utf-8")


def make_xlsx(texts, header="text"):
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["id", header])
    for i, t in enumerate(texts):
        ws.append([f"REQ-{i + 1:03d}", t])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ---------------------------------------------------------------- fake embedder

_TOKEN_VECS = {}


def _tok_vec(tok):
    v = _TOKEN_VECS.get(tok)
    if v is None:
        rng = np.random.default_rng(zlib.crc32(tok.encode()))
        v = rng.standard_normal(384).astype(np.float32)
        _TOKEN_VECS[tok] = v
    return v


class FakeEmbedder:
    """Deterministic bag-of-words embedder with a gate to hold clustering mid-flight."""

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.fail = None  # set to an exception instance to make embedding fail
        self.calls = 0

    def hold(self):
        self.gate.clear()
        self.entered.clear()

    def release(self):
        self.gate.set()

    def __call__(self, texts, *a, progress_callback=None, **k):
        self.calls += 1
        if not self.gate.is_set():
            self.entered.set()
            self.gate.wait(8)
        if self.fail is not None:
            raise self.fail
        if not len(texts):
            return np.zeros((0, 384), dtype=np.float32)
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                out[i] += _tok_vec(tok)
            n = np.linalg.norm(out[i])
            out[i] = out[i] / n if n else out[i]
        if progress_callback:
            progress_callback(len(texts), len(texts))
        return out

    def domain(self, texts, enriched_texts=None, config=None, progress_callback=None, **k):
        merged = [e if (enriched_texts and e) else t for t, e in zip(
            texts, enriched_texts or [None] * len(texts))]
        return self(merged, progress_callback=progress_callback)


# Modules that imported the embedding functions by name. Update for a new design.
EMBED_TARGETS = [
    "core.embeddings.generate_embeddings",
    "core.pipeline.generate_embeddings",
    "core.feedback_bridge.generate_embeddings",
    "services.refinement_service.generate_embeddings",
    "services.active_learning_service.generate_embeddings",
    "services.dependency_service.generate_embeddings",
]
DOMAIN_TARGETS = [
    "core.domain_embeddings.generate_domain_embeddings",
    "core.pipeline.generate_domain_embeddings",
]


@pytest.fixture
def embedder(monkeypatch):
    import importlib

    e = FakeEmbedder()
    for target, fn in [(t, e) for t in EMBED_TARGETS] + [(t, e.domain) for t in DOMAIN_TARGETS]:
        mod, attr = target.rsplit(".", 1)
        try:
            monkeypatch.setattr(importlib.import_module(mod), attr, fn, raising=False)
        except ImportError:
            pass
    monkeypatch.delenv("REDIS_URL", raising=False)
    return e


# ---------------------------------------------------------------- fake LLM HTTP server

def llm_json(**over):
    d = {
        "expanded_text": "Expanded requirement text.",
        "domain_terms": ["alpha"],
        "functional_intent": "do the thing",
        "mentioned_components": ["unit"],
        "assumptions": [],
        "confidence": 0.9,
        "warnings": [],
    }
    d.update(over)
    return d


def chat(content):
    return {"choices": [{"message": {"content": content}}]}


class FakeLLM:
    """Scripted OpenAI-compatible server. responses: list of (status, body_str);
    the last entry repeats. `handler` (optional) gets the request dict and returns
    (status, body_str) to override."""

    def __init__(self):
        self.requests = []
        self.responses = [(200, json.dumps(chat(json.dumps(llm_json()))))]
        self.handler = None
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                try:
                    parsed = json.loads(body)
                except Exception:
                    parsed = None
                req = {"path": self.path, "headers": dict(self.headers), "json": parsed}
                outer.requests.append(req)
                idx = min(len(outer.requests) - 1, len(outer.responses) - 1)
                status, text = outer.responses[idx]
                if outer.handler:
                    status, text = outer.handler(req)
                data = text.encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def prompts(self):
        out = []
        for r in self.requests:
            for m in (r["json"] or {}).get("messages", []):
                out.append(m.get("content", ""))
        return "\n".join(out)

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def fake_llm(monkeypatch):
    srv = FakeLLM()
    monkeypatch.setenv("REQCLUSTER_LLM_BASE_URL", srv.url)
    monkeypatch.setenv("REQCLUSTER_LLM_API_KEY", "test-key")
    monkeypatch.setenv("REQCLUSTER_LLM_MODEL", "fake-model")
    monkeypatch.setenv("REQCLUSTER_LLM_MAX_RETRIES", "2")
    monkeypatch.setenv("REQCLUSTER_LOCAL_LLM_URL", srv.url + "/generate")
    yield srv
    srv.close()


# ---------------------------------------------------------------- the adapter

def _body(resp):
    try:
        return resp.json()
    except Exception:
        return resp.text


class Run:
    """A cluster request in flight (legacy POST blocks, so it runs in a thread)."""

    def __init__(self, api, sid, params):
        self.api, self.sid = api, sid
        self.resp = None
        self.t = threading.Thread(target=self._go, args=(params,), daemon=True)
        self.t.start()

    @property
    def job_id(self):
        b = self.start_body
        return b.get("job_id") or b.get("id") if isinstance(b, dict) else None

    def _go(self, params):
        self.resp = self.api.c.post("/api/cluster", json={"session_id": self.sid, **params})

    @property
    def start_status(self):
        self.t.join(60)
        return self.resp.status_code if self.resp is not None else None

    @property
    def start_body(self):
        self.t.join(60)
        return _body(self.resp)


class LegacyApi:
    TERMINAL = {"done", "error"}
    GETS = {  # resource name -> (path, param style)
        "session": "/api/sessions/{sid}",
        "clusters": "/api/clusters?session_id={sid}",
        "requirements": "/api/requirements?session_id={sid}",
        "graph": "/api/graph?session_id={sid}",
        "metrics": "/api/metrics?session_id={sid}",
        "suggestions": "/api/suggestions?session_id={sid}",
        "audit": "/api/suggestions/audit?session_id={sid}",
        "feedback_queue": "/api/feedback/queue?session_id={sid}",
        "constraints": "/api/feedback/constraints?session_id={sid}",
        "feedback_export": "/api/feedback/export?session_id={sid}",
        "dependencies": "/api/dependencies?session_id={sid}",
        "active_learning_queue": "/api/active-learning/queue?session_id={sid}",
        "quality_history": "/api/quality/history?session_id={sid}",
        "enrich_status": "/api/enrich/status/{sid}",
        "enrich_results": "/api/enrich/results?session_id={sid}",
        "progress": "/api/progress/{sid}",
    }

    def __init__(self, client):
        self.c = client

    # -- plumbing
    def _r(self, resp):
        return resp.status_code, _body(resp)

    def get(self, name, sid):
        return self._r(self.c.get(self.GETS[name].format(sid=sid)))

    @staticmethod
    def error_message(body):
        if not isinstance(body, dict):
            return None
        d = body.get("detail", body.get("message"))
        if isinstance(d, str):
            return d or None
        if isinstance(d, list):
            m = " ".join(str(x.get("msg", "")) for x in d if isinstance(x, dict)).strip()
            return m or None
        if isinstance(d, dict) and isinstance(d.get("message"), str):
            return d["message"]
        return None

    # -- sessions
    def upload(self, filename, content):
        r = self.c.post("/api/upload", files={"file": (filename, content)})
        return self._r(r)

    def upload_texts(self, texts, **kw):
        st, b = self.upload("reqs.csv", make_csv(texts, **kw))
        assert st == 200, (st, b)
        return b["session_id"]

    def session(self, sid):
        return self.get("session", sid)

    # -- clustering
    def cluster_start(self, sid, **params):
        return Run(self, sid, params)

    def progress(self, sid):
        st, b = self.get("progress", sid)
        raw = b if isinstance(b, dict) else {}
        step = raw.get("status") or raw.get("state") or raw.get("step") or "idle"
        state = {"done": "done", "succeeded": "done", "error": "error", "failed": "error",
                 "idle": "idle"}.get(step, "running")
        return {"http": st, "state": state, "message": raw.get("message"), "raw": raw}

    def wait(self, run, timeout=90):
        """-> {"http", "state": done|error|rejected, "message", "body"}"""
        http = run.start_status
        if http is None or http >= 300 or http == 204:
            return {"http": http, "state": "rejected", "message": self.error_message(run.start_body),
                    "body": run.start_body}
        end = time.time() + timeout
        while time.time() < end:
            _, s = self.session(run.sid)
            if isinstance(s, dict) and s.get("status") in self.TERMINAL:
                p = self.progress(run.sid)
                return {"http": http, "state": s["status"], "message": p["message"], "body": s}
            time.sleep(0.1)
        return {"http": http, "state": "timeout", "message": None, "body": None}

    def cluster(self, sid, **params):
        return self.wait(self.cluster_start(sid, **params))

    def job(self, job_id):
        raise NotImplementedError("legacy API has no job resource")

    def clusters(self, sid):
        return self.get("clusters", sid)

    def cluster_detail(self, sid, cluster_id):
        return self._r(self.c.get(f"/api/cluster/{cluster_id}?session_id={sid}"))

    def requirements(self, sid, **q):
        """-> (http, items, total_count|None)"""
        r = self.c.get("/api/requirements", params={"session_id": sid, **q})
        t = r.headers.get("x-total-count")
        return r.status_code, _body(r), int(t) if t else None

    def all_requirements(self, sid):
        st, body, _ = self.requirements(sid, page=1, page_size=1000)
        assert st == 200, (st, body)
        return sorted(body, key=lambda r: r["id"])

    def labels(self, sid):
        return [r["cluster_id"] for r in self.all_requirements(sid)]

    def recluster_constrained(self, sid):
        return self._r(self.c.post("/api/cluster/constrained", json={"session_id": sid}))

    # -- feedback / refinement
    def move(self, sid, requirement_id, new_cluster_id, **kw):
        return self._r(self.c.post("/api/feedback/submit", json={
            "session_id": sid, "requirement_id": requirement_id,
            "new_cluster_id": new_cluster_id, **kw}))

    def review_feedback(self, sid, feedback_id, status="approved"):
        return self._r(self.c.post("/api/feedback/review", json={
            "session_id": sid, "feedback_id": feedback_id, "status": status}))

    def suggest(self, sid, **kw):
        return self._r(self.c.post("/api/suggestions/generate", json={"session_id": sid, **kw}))

    def suggestions(self, sid, status=None):
        r = self.c.get("/api/suggestions", params={"session_id": sid, **({"status": status} if status else {})})
        return self._r(r)

    def apply_suggestion(self, sid, suggestion_id, action="accept"):
        return self._r(self.c.post("/api/suggestions/apply", json={
            "session_id": sid, "suggestion_id": suggestion_id, "action": action}))

    # -- dependencies
    def dependencies_generate(self, sid, **kw):
        return self._r(self.c.post("/api/dependencies/generate", json={"session_id": sid, **kw}))

    def dependencies(self, sid):
        return self.get("dependencies", sid)

    def explain(self, sid, source, target, **kw):
        return self._r(self.c.post(f"/api/dependencies/{sid}/explain", json={
            "source_index": source, "target_index": target, **kw}))

    # -- enrichment
    def enrich(self, sid, **kw):
        return self._r(self.c.post("/api/enrich", json={"session_id": sid, **kw}))

    # -- exports
    def export(self, sid, fmt):
        r = self.c.get(f"/api/export/{fmt}", params={"session_id": sid})
        return r.status_code, r.content, r.headers


@pytest.fixture
def api(tmp_path, embedder, monkeypatch):
    """Swap point: build the app under test against a temp DB, return an adapter."""
    from api import routes
    from models.database import Base, get_db

    engine = create_engine(f"sqlite:///{tmp_path / 'acc.db'}",
                           connect_args={"check_same_thread": False})
    Session = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)

    def override_get_db():
        db = Session()
        try:
            yield db
        finally:
            db.close()

    app = FastAPI()
    app.include_router(routes.router, prefix="/api")
    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[routes.get_db] = override_get_db
    monkeypatch.setattr(routes, "SessionLocal", Session)
    routes.pipeline_progress.clear()
    routes.progress_cache.clear()
    monkeypatch.setenv("REQCLUSTER_LLM_PROVIDER", "mock")
    with TestClient(app, raise_server_exceptions=False) as client:
        yield LegacyApi(client)
    engine.dispose()


@pytest.fixture
def clustered(api):
    """A session with 4 well-separated topics, clustered. -> sid"""
    sid = api.upload_texts(topic_texts())
    res = api.cluster(sid)
    assert res["state"] == "done", res
    return sid


@pytest.fixture
def dep_session(api):
    """Clustered session with a dependency tree that has at least one edge. -> sid"""
    sid = api.upload_texts(DEP_TEXTS)
    assert api.cluster(sid, min_cluster_size=2)["state"] == "done"
    st, b = api.dependencies_generate(sid, sim_threshold=0.0)
    assert st == 200 and b["edges"], (st, b)
    return sid
