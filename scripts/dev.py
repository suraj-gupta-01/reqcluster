"""Cross-platform launcher: everything needed to use the app locally.

Installs frontend deps if missing, starts Ollama when the local LLM provider is configured and it is
not running, starts the backend (reqcluster-api) and the frontend (Vite), opens the browser once
both answer, and stops everything it started on Ctrl+C or when either server exits.

Run: ./start.sh (Linux/macOS) or .\\start.ps1 (Windows); both install Python deps first.
Directly: uv run python scripts/dev.py [--no-browser]
Backend port: REQCLUSTER_PORT (default 8000).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = "http://localhost:5173"


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            if sys.platform == "win32":  # terminate() kills npm.cmd but leaves node/vite running
                subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
            else:
                p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def _up(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except Exception:  # connection refused, timeout, HTTP error: not ready
        return False


def _ollama(procs: list[subprocess.Popen]) -> None:
    """Start Ollama only when the configured provider is the local one and it is not already up."""
    try:
        from reqcluster.config import get_settings  # honours .env

        s = get_settings()
    except Exception as e:  # bad .env value: the backend will report it on start
        print(f"(skipping Ollama check: {e})")
        return
    if s.llm_provider != "local":
        return
    tags = s.local_llm_url.rstrip("/") + "/api/tags"
    if _up(tags):
        return
    exe = shutil.which("ollama")
    if not exe:
        print("LLM provider is 'local' but Ollama is not installed; LLM features fall back to "
              "template text. Install from https://ollama.com or set REQCLUSTER_LLM_PROVIDER=mock.")
        return
    print("Starting Ollama ...")
    procs.append(subprocess.Popen([exe, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
    for _ in range(30):
        if _up(tags):
            return
        time.sleep(0.5)
    print("Ollama did not answer within 15 s; continuing (requests will retry).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--no-browser", action="store_true", help="do not open the app in a browser")
    args = ap.parse_args()

    env = {**os.environ, "REQCLUSTER_HOST": "127.0.0.1"}
    port = env.setdefault("REQCLUSTER_PORT", "8000")
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm") or shutil.which("npm")
    if not npm:
        print("npm not found on PATH (install Node.js 20+)", file=sys.stderr)
        return 1
    if not (ROOT / "frontend" / "node_modules").exists():
        print("Installing frontend dependencies (first run) ...")
        if subprocess.run([npm, "ci"], cwd=ROOT / "frontend").returncode:
            return 1

    procs: list[subprocess.Popen] = []
    _ollama(procs)
    procs += [
        subprocess.Popen(["uv", "run", "--no-sync", "reqcluster-api"], cwd=ROOT, env=env),
        subprocess.Popen([npm, "run", "dev"], cwd=ROOT / "frontend", env=env),
    ]
    servers = procs[-2:]
    health = f"http://127.0.0.1:{port}/health"
    print(f"Backend  http://127.0.0.1:{port}  (API docs: /docs)\nFrontend {FRONTEND}\nCtrl+C to stop.")
    opened = args.no_browser
    try:
        while all(p.poll() is None for p in servers):
            if not opened and _up(health) and _up(FRONTEND):  # model load can take ~20 s
                print(f"Ready: {FRONTEND}")
                webbrowser.open(FRONTEND)
                opened = True
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _stop(procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
