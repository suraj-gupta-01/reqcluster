"""9. Exports."""
import json
import xml.etree.ElementTree as ET

import pytest

from .conftest import topic_texts

pytestmark = pytest.mark.acceptance

NASTY = [
    "Control \x01 char and \x08 backspace and \x1b escape shall not break XML export",
    "Fish & chips <ok shall be served with &amp; literal and \"quotes\" and 'apostrophes'",
    "The rocket \U0001F680 emoji shall survive the export round trip intact",
    "CDATA ]]> terminator shall not break the document structure here",
]


@pytest.fixture
def nasty_session(api):
    sid = api.upload_texts(topic_texts(6) + NASTY)
    assert api.cluster(sid, min_cluster_size=3)["state"] == "done"
    reqs = api.all_requirements(sid)
    assert api.move(sid, reqs[0]["id"], -1)[0] == 200     # guaranteed noise (-1) row
    return sid


def _ids(root):
    seen = []
    for el in root.iter():
        for k, v in el.attrib.items():
            if k == "IDENTIFIER" or k == "id" or k.endswith("}id"):
                seen.append(v)
    return seen


@pytest.mark.parametrize("fmt", ["reqif", "sysml"])
def test_xml_exports_wellformed_with_nasty_content(api, nasty_session, fmt):
    st, content, _ = api.export(nasty_session, fmt)
    assert st == 200
    root = ET.fromstring(content)  # raises on malformed XML
    text = "".join(root.itertext())
    assert "\U0001F680" in text
    assert "Fish &" in text


@pytest.mark.parametrize("fmt", ["reqif", "sysml"])
def test_xml_exports_have_unique_identifiers_with_noise(api, nasty_session, fmt):
    st, content, _ = api.export(nasty_session, fmt)
    assert st == 200
    ids = _ids(ET.fromstring(content))
    assert ids, "no identifiers found"
    dupes = {i for i in ids if ids.count(i) > 1}
    assert not dupes, dupes
    assert all(i and i.strip() and i != "None" and "None" not in i for i in ids)


def test_noise_requirements_present_in_exports(api, nasty_session):
    st, content, _ = api.export(nasty_session, "reqif")
    assert len(ET.fromstring(content).findall(".//{*}SPEC-OBJECT")) >= 24 + len(NASTY)


def test_jama_export_is_valid_json(api, nasty_session):
    st, content, _ = api.export(nasty_session, "jama")
    assert st == 200
    json.loads(content)


def test_csv_export_roundtrips_special_text(api, nasty_session):
    import csv, io
    st, content, _ = api.export(nasty_session, "csv")
    assert st == 200
    rows = list(csv.reader(io.StringIO(content.decode("utf-8-sig"))))
    assert len(rows) - 1 == 24 + len(NASTY)
    assert any("\U0001F680" in c for r in rows for c in r)


def test_pdf_export_survives_special_chars(api, nasty_session):
    st, content, _ = api.export(nasty_session, "pdf")
    assert st == 200 and content[:4] == b"%PDF"


@pytest.mark.parametrize("fmt", ["reqif", "sysml", "jama", "csv"])
def test_export_unclustered_session_not_500(api, fmt):
    sid = api.upload_texts(topic_texts(2))
    st, content, _ = api.export(sid, fmt)
    assert st in (200, 409), st


def test_export_unknown_format_400(api):
    sid = api.upload_texts(topic_texts(2))
    st, content, _ = api.export(sid, "docx")
    assert st == 400
