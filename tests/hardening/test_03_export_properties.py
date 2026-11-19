import csv
import io
import json
import xml.etree.ElementTree as ET
from collections import Counter

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sqlalchemy import select, update

from .conftest import run_cluster, running_app, topic_texts, upload

pytestmark = pytest.mark.hardening
S = settings(max_examples=60, derandomize=True, deadline=None, suppress_health_check=list(HealthCheck))
FORMATS = ["reqif", "sysml", "jama", "csv", "pdf"]
SPICY = st.sampled_from(["]]>", "&", "<", ">", "\"'", "&amp;", "&#0;", "😀", "=1+1", "+1", "-1", "@x", "\t=x",
                         "\r=x", "\x00", "\x0b", "\x1f", "￾", "￿", " ", "퟿", "<![CDATA[", "--"])
txt = st.lists(st.one_of(SPICY, st.text(max_size=30)), min_size=1, max_size=6).map("".join)


@pytest.fixture(scope="module")
def sess(tmp_path_factory):
    with running_app(tmp_path_factory.mktemp("exp")) as (c, _):
        sid = upload(c, topic_texts(5))
        assert run_cluster(c, sid)["status"] == "succeeded"
        yield c, sid


def rewrite(sid, texts, label):
    from reqcluster.db import session_scope
    from reqcluster.models import ClusterRow, RequirementRow, SessionRow
    with session_scope() as s:
        rows = list(s.scalars(select(RequirementRow).where(RequirementRow.session_id == sid)
                              .order_by(RequirementRow.ordinal)))
        for i, r in enumerate(rows):
            r.text = texts[i % len(texts)]
            r.req_id = f"{texts[i % len(texts)][:20]}#{i}"
            r.attributes = {"k" + texts[0][:5]: texts[-1]}
        s.execute(update(ClusterRow).values(label=label, keywords=[label, "a&b"]))
        s.execute(update(SessionRow).where(SessionRow.id == sid).values(name=label))
        return [r.text for r in rows]


def attr_ids(root):
    out = []
    for e in root.iter():
        for k, v in e.attrib.items():
            if k.split("}")[-1] in ("IDENTIFIER", "id", "ID"):
                out.append(v)
    return out


@given(st.lists(txt, min_size=1, max_size=5), txt)
@S
def test_exports_hold_for_hostile_text(sess, texts, label):
    c, sid = sess
    stored = rewrite(sid, texts, label)
    for fmt in FORMATS:
        r = c.get(f"/api/sessions/{sid}/export/{fmt}")
        assert r.status_code == 200, (fmt, r.status_code, r.text[:200], texts, label)
        assert r.content
        if fmt in ("reqif", "sysml"):
            root = ET.fromstring(r.content)
            ids = attr_ids(root)
            assert ids and len(ids) == len(set(ids)), (fmt, [k for k, n in Counter(ids).items() if n > 1])
        elif fmt == "jama":
            json.loads(r.content)
        elif fmt == "pdf":
            assert r.content.startswith(b"%PDF")
        else:
            text = r.content.decode("utf-8-sig")
            rows = list(csv.reader(io.StringIO(text, newline="")))
            head, body = rows[0], rows[1:]
            assert len(body) == len(stored)
            ti = head.index("text")
            for row in body:
                for cellv in row:
                    if cellv and cellv[0] in "=+-@\t\r":
                        try:
                            float(cellv)
                        except ValueError:
                            raise AssertionError(f"formula-injectable cell {cellv!r}") from None
            got = [row[ti] for row in body]
            for g, w in zip(got, stored, strict=True):
                assert g == w or g == "'" + w, (g, w)


def test_export_unknown_format_and_unclustered(app):
    c, _ = app
    sid = upload(c, topic_texts(2))
    assert c.get(f"/api/sessions/{sid}/export/nope").status_code in (400, 404)
    for fmt in FORMATS:  # not clustered yet: must be a clean 4xx or a valid file, never 500
        assert c.get(f"/api/sessions/{sid}/export/{fmt}").status_code < 500
    assert c.get("/api/sessions/999999/export/csv").status_code == 404
