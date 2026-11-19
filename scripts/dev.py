"""Cross-platform dev launcher: backend (reqcluster-api) + frontend (Vite).

Run: uv run python scripts/dev.py
Backend port: REQCLUSTER_PORT (default 8000). Stops both on Ctrl+C or when either exits.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _stop(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        if p.poll() is None:
            p.terminate()
    for p in procs:
        try:
            p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            p.kill()
            p.wait()


def main() -> int:
    env = {**os.environ, "REQCLUSTER_HOST": "127.0.0.1"}
    port = env.setdefault("REQCLUSTER_PORT", "8000")
    npm = shutil.which("npm.cmd" if sys.platform == "win32" else "npm") or shutil.which("npm")
    if not npm:
        print("npm not found on PATH", file=sys.stderr)
        return 1
    print(f"Backend  http://127.0.0.1:{port}  (docs: /docs)\nFrontend http://localhost:5173\nCtrl+C to stop.")
    procs = [
        subprocess.Popen(["uv", "run", "reqcluster-api"], cwd=ROOT, env=env),
        subprocess.Popen([npm, "run", "dev"], cwd=ROOT / "frontend", env=env),
    ]
    try:
        while all(p.poll() is None for p in procs):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        _stop(procs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
