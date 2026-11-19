import csv
import io
import json
import xml.etree.ElementTree as ET

import pytest

from reqcluster.db import init_db, session_scope
from reqcluster.errors import Invalid, NotFound
from reqcluster.export import export
from reqcluster.export.bundle import build, xml_id, xml_safe
from reqcluster.models import AssignmentRow, ClusterRow, RequirementRow, RunRow, SessionRow

NASTY = ["Ctl \x01 \x08 \x1b end", "Fish & chips <ok \"q\" 'a'", "rocket \U0001F680", "CDATA ]]> end"]


def make(texts, cids=None, req_ids=None):
    """cids None = unclustered; else per-requirement cluster id (-1 noise)."""
    init_db()
    with session_scope() as s:
        sess = SessionRow(name="Demo & <co>", n_requirements=len(texts))
        s.add(sess)
        s.flush()
        rows = []
        for i, t in enumerate(texts):
            rid = req_ids[i] if req_ids else f"R{i}"
            r = RequirementRow(session_id=sess.id, ordinal=i, req_id=rid, text=t)
            s.add(r)
            rows.append(r)
        s.flush()
        if cids is not None:
            run = RunRow(session_id=sess.id, revision=1, kind="cluster")
            s.add(run)
            s.flush()
            sess.current_run_id = run.id
            for c in sorted({c for c in cids if c != -1}):
                s.add(ClusterRow(run_id=run.id, cluster_id=c, size=cids.count(c),
                                 label=f"Topic {c}", keywords=["k1", "k2"]))
            for r, c in zip(rows, cids, strict=True):
                s.add(AssignmentRow(run_id=run.id, requirement_id=r.id, cluster_id=c))
        return sess.id


def ids(root):
    return [v for el in root.iter() for k, v in el.attrib.items()
            if k in ("IDENTIFIER",) or k.endswith("}id")]


def test_bundle_basic_and_noise():
    sid = make(["a", "b", "c"], [0, 0, -1])
    b = build(sid)
    assert [r.cluster_label for r in b.requirements] == ["Topic 0", "Topic 0", "Noise"]
    assert b.requirements[2].is_noise and [c.id for c in b.clusters] == [0]


def test_unclustered_and_missing_req_id():
    sid = make(["a", "b"], None, req_ids=["", "  "])
    b = build(sid)
    assert {r.cluster_label for r in b.requirements} == {"Unclustered"}
    assert [r.req_id for r in b.requirements] == ["REQ-00000", "REQ-00001"]
    for fmt in ("reqif", "sysml", "jama", "csv", "pdf"):
        assert export(sid, fmt)[0]


def test_all_noise():
    sid = make(["a", "b"], [-1, -1])
    for fmt in ("reqif", "sysml"):
        ET.fromstring(export(sid, fmt)[0])
    assert len(json.loads(export(sid, "jama")[0])["items"]) == 2


def test_xml_safe():
    assert xml_safe("a\x00b\x0bc\ud800d￾e\t\n\r") == "abcde\t\n\r"
    seen = set()
    a, b2 = xml_id("_x", "a b", seen), xml_id("_x", "a b", seen)
    assert a != b2 and " " not in a


@pytest.mark.parametrize("fmt", ["reqif", "sysml"])
def test_nasty_xml_wellformed_unique(fmt):
    sid = make(NASTY + ["x"], [0, 0, -1, 0, 0], req_ids=["D", "D", "D", "a b", "\x01"])
    content, media, name = export(sid, fmt)
    root = ET.fromstring(content)
    text = "".join(root.itertext())
    assert "\U0001F680" in text and "Fish &" in text and "]]>" in text and "\x01" not in text
    got = ids(root)
    assert got and len(got) == len(set(got))


def test_reqif_structure():
    sid = make(["a", "b", "c"], [0, 1, -1])
    ns = "{http://www.omg.org/spec/ReqIF/20110401/reqif.xsd}"
    root = ET.fromstring(export(sid, "reqif")[0])
    assert len(root.findall(f".//{ns}SPEC-OBJECT")) == 5  # 3 reqs + 2 headings
    assert len(root.findall(f".//{ns}SPEC-HIERARCHY")) == 5


def test_sysml_structure_and_edges():
    sid = make(["a", "b", "c"], [0, 1, -1])
    b = build(sid)
    b.edges.append(("R0", "R1", "data", 1.0))
    from reqcluster.export.xmi import XMI_NS, export_sysml
    root = ET.fromstring(export_sysml(b))
    def q(t):
        return root.findall(f".//packagedElement[@{{{XMI_NS}}}type='{t}']")

    assert (len(q("uml:Package")), len(q("uml:Class")), len(q("uml:Dependency"))) == (2, 3, 1)


def test_jama_uses_settings(monkeypatch):
    monkeypatch.setenv("REQCLUSTER_JAMA_ITEM_TYPE_ID", "77")
    sid = make(["a"], [0])
    out = json.loads(export(sid, "jama")[0])
    assert out["format"] == "jama" and out["meta"]["item_count"] == 1
    assert out["items"][0]["documentKey"] == "R0"
    assert out["items"][0]["itemType"] == 77


def test_csv_roundtrip_and_formula_guard():
    sid = make(["=cmd()", "-5", "+1+1", "@x", "-1.5", "multi\nline, \"q\" \U0001F680"], [0] * 6)
    data = export(sid, "csv")[0]
    assert data.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    texts = [r[1] for r in rows[1:]]
    assert texts == ["'=cmd()", "-5", "'+1+1", "'@x", "-1.5", "multi\nline, \"q\" \U0001F680"]


def test_pdf_valid_and_empty_clusters():
    sid = make(NASTY, [0, 0, -1, 0])
    pdf = export(sid, "pdf")[0]
    assert pdf[:5] == b"%PDF-" and len(pdf) > 1000
    assert export(make(["a"], [-1]), "pdf")[0][:5] == b"%PDF-"


def test_errors():
    sid = make(["a"], None)
    with pytest.raises(Invalid):
        export(sid, "docx")
    with pytest.raises(NotFound):
        export(9999, "csv")


def test_csv_guards_attribute_header_names():
    from reqcluster.export.bundle import to_csv
    from reqcluster.ingest import parse
    from reqcluster.services import sessions
    from reqcluster.db import init_db

    init_db()
    res = parse(b"id,text,=cmd\nR1,The pump shall start within 2 seconds,x\n", "a.csv")
    sid = sessions.create(res, "a").id
    header = to_csv(build(sid)).decode("utf-8-sig").splitlines()[0]
    assert ",'=cmd" in header
