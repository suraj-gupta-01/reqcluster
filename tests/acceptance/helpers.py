"""Shared assertions for acceptance tests (contract-level, no internals)."""
import csv
import io
import re

NOISE = (-1, None)


def assert_coherent(api, sid, expect_n=None):
    """Cluster results must be self-consistent: noise flag vs cluster id vs counts."""
    reqs = api.all_requirements(sid)
    st, s = api.session(sid)
    assert st == 200 and s["status"] == "done", s
    st, clusters = api.clusters(sid)
    assert st == 200
    ids = {c["cluster_id"] for c in clusters}
    assert -1 not in ids and None not in ids, "noise must not be a cluster entry"
    if expect_n is not None:
        assert len(reqs) == expect_n
    n_noise = 0
    sizes = {}
    for r in reqs:
        if r["is_noise"]:
            n_noise += 1
            assert r["cluster_id"] in NOISE, r
        else:
            assert r["cluster_id"] is not None and r["cluster_id"] >= 0, r
            assert r["cluster_id"] in ids, r
            sizes[r["cluster_id"]] = sizes.get(r["cluster_id"], 0) + 1
    assert s["noise_count"] == n_noise
    assert s["total_clusters"] == len(clusters)
    for c in clusters:
        assert c["size"] == sizes.get(c["cluster_id"], 0), c
    assert sum(sizes.values()) + n_noise == len(reqs)
    return reqs, clusters


def csv_dangerous_cells(content):
    """Cells a spreadsheet would execute (start with = + - @ tab CR), excluding plain numbers."""
    text = content.decode("utf-8-sig") if isinstance(content, bytes) else content
    bad = []
    for row in csv.reader(io.StringIO(text)):
        for cell in row:
            if cell[:1] in ("=", "+", "-", "@", "\t", "\r") and not re.fullmatch(r"-?\d+(\.\d+)?", cell):
                bad.append(cell)
    return bad


def requirements_by_text(reqs):
    return {r["text"]: r for r in reqs}


def assert_json_error(api, status, body, expect_status=None):
    if expect_status is not None:
        assert status == expect_status, (status, body)
    msg = api.error_message(body)
    assert isinstance(msg, str) and msg.strip(), body
    assert "Traceback" not in msg and ".py" not in msg
