"""Hardening suite fixtures. Run with: uv run pytest -m hardening tests/hardening -q"""
import os
import contextlib
import io
import json
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.hardening

TERMINAL = {"succeeded", "failed", "cancelled"}
TOPICS = [
    ("brake", "pedal", "caliper", "hydraulic"),
    ("battery", "cell", "charger", "voltage"),
    ("display", "screen", "pixel", "backlight"),
    ("radio", "antenna", "signal", "frequency"),
]


def pytest_collection_modifyitems(items):
    for it in items:
        if "tests/hardening" in str(it.fspath).replace("\\", "/"):
            it.add_marker(pytest.mark.hardening)


def topic_texts(per_topic=10, tag=""):
    return [f"The {t[0]} {t[1]} {t[2]} {t[3]} subsystem shall report {t[0]} {t[3]} status "
            f"code {i}{tag} within {10 + i} milliseconds" for t in TOPICS for i in range(per_topic)]


def csv_bytes(texts):
    import csv
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["id", "text"])
    for i, t in enumerate(texts):
        w.writerow([f"REQ-{i + 1:04d}", t])
    return buf.getvalue().encode()


def _tok_vec(tok):
    rng = np.random.default_rng(zlib.crc32(tok.encode()))
    return rng.standard_normal(384).astype(np.float32)


class FakeEmbedder:
    """Bag-of-words embedder with a gate (hold) and failure injection (fail)."""
    name, revision, is_mock, dim = "fake", "1", True, 384

    def __init__(self):
        self.gate = threading.Event()
        self.gate.set()
        self.entered = threading.Event()
        self.fail = None
        self._cache = {}

    def hold(self):
        self.gate.clear()
        self.entered.clear()

    def release(self):
        self.gate.set()

    def encode(self, texts, batch_size=64):
        # only job threads are held, so app startup (warm-up encode) is not blocked
        if not self.gate.is_set() and threading.current_thread().name.startswith("job-"):
            self.entered.set()
            self.gate.wait(15)
        if self.fail is not None:
            raise self.fail
        out = np.zeros((len(texts), 384), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in str(t).lower().split():
                v = self._cache.get(tok)
                if v is None:
                    v = self._cache[tok] = _tok_vec(tok)
                out[i] += v
            n = np.linalg.norm(out[i])
            if n:
                out[i] /= n
        return out


def reset_all():
    from reqcluster import db, jobs
    from reqcluster.config import get_settings
    from reqcluster.llm.gateway import reset_gateway
    get_settings.cache_clear()
    jobs.reset_runner(wait=False)
    db.reset()
    reset_gateway()


@contextlib.contextmanager
def running_app(home, workers=0, fake_embedder=True, env=None):
    """Yield (client, embedder). Env is restored on exit."""
    mp = pytest.MonkeyPatch()
    mp.setenv("REQCLUSTER_HOME", str(home))
    mp.setenv("REQCLUSTER_EMBEDDING_BACKEND", "mock")
    mp.setenv("REQCLUSTER_LLM_PROVIDER", "mock")
    mp.setenv("REQCLUSTER_JOB_WORKERS", str(workers))
    mp.delenv("DATABASE_URL", raising=False)
    mp.delenv("REQCLUSTER_DATABASE_URL", raising=False)
    test_db = os.environ.get("REQCLUSTER_TEST_DATABASE_URL")
    if test_db:  # opt-in: run against a real server DB (e.g. Postgres) with a fresh schema per app
        mp.setenv("DATABASE_URL", test_db)
    for k, v in (env or {}).items():
        mp.setenv(k, str(v))
    emb = FakeEmbedder()
    if fake_embedder:
        mp.setattr("reqcluster.embeddings.get_embedder", lambda: emb)
    reset_all()
    if test_db:
        from reqcluster import db, models
        models.Base.metadata.drop_all(db.get_engine())
    from reqcluster.app import create_app
    try:
        with TestClient(create_app(), raise_server_exceptions=False) as c:
            yield c, emb
    finally:
        emb.release()
        reset_all()
        mp.undo()


@pytest.fixture
def app(tmp_path):
    with running_app(tmp_path) as ca:
        yield ca


@pytest.fixture
def app_async(tmp_path):
    with running_app(tmp_path, workers=1) as ca:
        yield ca


@pytest.fixture(scope="module")
def fuzz_app(tmp_path_factory):
    with running_app(tmp_path_factory.mktemp("fuzz")) as ca:
        yield ca


def upload(c, texts, name="r.csv"):
    r = c.post("/api/sessions", files={"file": (name, csv_bytes(texts))})
    assert r.status_code == 200, r.text
    return r.json()["session"]["id"]


def wait_job(c, jid, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in TERMINAL:
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {jid} not terminal after {timeout}s: {j}")


def run_cluster(c, sid, **params):
    r = c.post(f"/api/sessions/{sid}/cluster", json=params)
    assert r.status_code == 202, r.text
    return wait_job(c, r.json()["id"])


def is_envelope(r):
    try:
        b = r.json()
    except Exception:
        return False
    return isinstance(b, dict) and set(b) >= {"error", "message"}


class FakeLLM:
    """Scripted OpenAI-compatible server. mode: ok | 401 | 500 | garbage | html."""

    def __init__(self, mode="ok"):
        self.mode = mode
        self.hits = 0
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.hits += 1
                m = outer.mode
                status, body = 200, ""
                if m == "401":
                    status, body = 401, json.dumps({"error": {"message": "bad key"}})
                elif m == "500":
                    status, body = 500, "boom"
                elif m == "garbage":
                    body = json.dumps({"choices": [{"message": {"content": "}{ not json ]]"}}]})
                elif m == "html":
                    body = "<html>gateway</html>"
                else:
                    body = json.dumps({"choices": [{"message": {"content": json.dumps({
                        "expanded_text": "Expanded requirement text.", "domain_terms": ["a"],
                        "functional_intent": "x", "mentioned_components": [], "assumptions": [],
                        "confidence": 0.9, "warnings": []})}}]})
                data = body.encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()
