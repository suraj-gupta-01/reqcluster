import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_nginx_body_limit_covers_max_upload():
    m = re.search(r"client_max_body_size\s+(\d+)m;", (ROOT / "nginx.conf").read_text())
    assert m
    default = re.search(r'"REQCLUSTER_MAX_UPLOAD_MB",\s*(\d+)', (ROOT / "backend/reqcluster/config.py").read_text())
    assert int(m.group(1)) >= int(default.group(1))
