#!/usr/bin/env python
"""Serve the interactive Dependency Engine Hardening Demo UI.

Usage:
    uv run python scripts/serve_demo_ui.py [--port 8080]
"""

import argparse
import json
import urllib.request
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# Preload spacy before any torch import to avoid Windows DLL conflict
try:
    import spacy
    _nlp_preloaded = spacy.load("en_core_web_sm", disable=["ner"])
except Exception:
    _nlp_preloaded = None

ROOT = Path(__file__).resolve().parents[1]

from reqcluster.config import get_settings  # noqa: E402
from reqcluster.llm.gateway import get_gateway  # noqa: E402
from reqcluster.llm.narrate import explain_edge  # noqa: E402
from reqcluster.ml.dependency import extract as extractor  # noqa: E402
from reqcluster.ml.dependency.extract import extract_artifacts, spacy_available  # noqa: E402

if _nlp_preloaded is not None:
    extractor._NLP_CACHE["en_core_web_sm"] = _nlp_preloaded

STATIC_DIR = ROOT / "scripts" / "demo_ui"


class DemoAPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/status":
            self.handle_status()
        else:
            super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get("Content-Length", 0))
        post_body = self.rfile.read(content_len).decode("utf-8")
        try:
            req_data = json.loads(post_body) if post_body else {}
        except Exception:
            req_data = {}

        path = self.path.split("?")[0]

        if path == "/api/extract":
            self.handle_extract(req_data)
        elif path == "/api/explain":
            self.handle_explain(req_data)
        elif path == "/api/status":
            self.handle_status()
        else:
            self.send_error(404, "Endpoint not found")

    def handle_status(self):
        cfg = get_settings()
        provider, base_url, model = cfg.llm_provider, cfg.llm_base_url, cfg.llm_model

        # Test LM Studio
        llm_online = False
        try:
            req = urllib.request.Request(f"{base_url}/models")
            with urllib.request.urlopen(req, timeout=1.5) as resp:
                resp.read()
                llm_online = True
        except Exception:
            llm_online = False

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "spacy_available": spacy_available(),
            "llm_provider": provider,
            "llm_model": model,
            "llm_base_url": base_url,
            "llm_online": llm_online,
        }).encode("utf-8"))

    def handle_extract(self, data):
        sentence = data.get("sentence", "").strip()
        if not sentence:
            self.send_error(400, "Sentence is required")
            return

        regex_res = extract_artifacts([sentence], ["TEST-001"], use_spacy=False)[0]
        spacy_res = extract_artifacts([sentence], ["TEST-001"], use_spacy=True)[0] if spacy_available() else None

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "sentence": sentence,
            "regex": {
                "produced": sorted(list(regex_res.produced)),
                "consumed": sorted(list(regex_res.consumed)),
            },
            "spacy": {
                "produced": sorted(list(spacy_res.produced)) if spacy_res else [],
                "consumed": sorted(list(spacy_res.consumed)) if spacy_res else [],
            } if spacy_res else None,
            "spacy_active": spacy_available(),
        }).encode("utf-8"))

    def handle_explain(self, data):
        source_id = data.get("source_id", "REQ-001")
        target_id = data.get("target_id", "REQ-002")
        source_text = data.get("source_text", "")
        target_text = data.get("target_text", "")
        relation = data.get("relation", "data")
        evidence = data.get("evidence", {})
        confidence_tier = data.get("confidence_tier", "High")
        rationale = data.get("rationale", "")
        use_mock = data.get("use_mock", False)

        result = explain_edge(
            get_gateway(),
            source_text,
            target_text,
            relation,
            evidence,
            source_id=source_id,
            target_id=target_id,
            confidence_tier=confidence_tier,
            rationale=rationale,
            provider="mock" if use_mock else None,
        )
        # The gateway falls back to the deterministic template on LLM failure.
        resp_payload = {
            "success": result.error is None,
            "provider": result.provider,
            "explanation": result.output,
        }
        if result.error:
            resp_payload["error"] = result.error
            resp_payload["fallback_provider"] = "mock"

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(resp_payload).encode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="ReqCluster Hardening Demo UI Server")
    parser.add_argument("--port", type=int, default=8090, help="Port to serve UI (default: 8090)")
    args = parser.parse_args()

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    server_address = ("", args.port)
    httpd = HTTPServer(server_address, DemoAPIHandler)

    print("\n" + "=" * 65)
    print("  ReqCluster Dependency Hardening Demo UI")
    print(f"  Running at: http://localhost:{args.port}")
    print("  Local LM Studio bridge: Active")
    print("=" * 65 + "\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")


if __name__ == "__main__":
    main()
