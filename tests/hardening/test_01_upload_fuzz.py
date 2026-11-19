import csv
import datetime
import io

import openpyxl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from .conftest import is_envelope

pytestmark = pytest.mark.hardening
S = settings(max_examples=120, derandomize=True, deadline=None,
             suppress_health_check=list(HealthCheck))

NASTY = ["‮RTL‬", "שלום العربية", "😀🚀👨‍👩‍👧",
         "zero​width‍join﻿er", "퟿￿\U0010ffff", "=1+1", "+cmd|' /C calc'!A0",
         "-2+3", "@SUM(A1)", "\tTAB", "]]>", "<b>tag</b>", "&amp; & <", "x" * 12000, "12345678",
         "'quoted'", '"dq"', "a\nb\r\nc", "\x00\x01bell\x07", "ſtraſſe STRASSE Straße", "İI ıi"]
cell = st.one_of(st.sampled_from(NASTY), st.text(max_size=80),
                 st.text(alphabet="shall the system 0123456789,;\t|\"'\n ", max_size=120),
                 st.integers().map(str))
HEADERS = [["id", "text"], ["ID", "Requirement"], ["﻿text"], ["text", "text"], ["id", "id", "text"],
           ["", "text", ""], [" Description "], ["foo", "bar"], ["text"] * 30, ["requirement text", "shall"],
           ["REQ ID", "Text"], ["req_id", "requirement"]]
NAMES = ["a.csv", "a.CSV", "a.xlsx", "a.xls", "a.txt", "a", "", "a.csv.exe", "../../x.csv", "a.xlsx.csv"]


COUNT = {}


def check(c, data, name):
    r = c.post("/api/sessions", files={"file": (name, data)})
    COUNT[r.status_code] = COUNT.get(r.status_code, 0) + 1
    ok = (200, 400, 413) + ((422,) if name == "" else ())  # empty filename = not a file part
    assert r.status_code in ok, (r.status_code, r.text[:300], name, data[:200])
    if r.status_code != 200:
        assert is_envelope(r), r.text[:200]
        return
    sid = r.json()["session"]["id"]
    items, page = [], 1
    while True:
        g = c.get(f"/api/sessions/{sid}/requirements", params={"page": page, "page_size": 1000})
        assert g.status_code == 200
        items += g.json()
        if len(items) >= int(g.headers["x-total-count"]) or not g.json():
            break
        page += 1
    assert items
    texts = [i["text"] for i in items]
    assert all(t.strip() for t in texts)
    assert len({t.casefold() for t in texts}) == len(texts), "duplicate texts (casefold)"
    assert all(len(t) <= 10_000 for t in texts)
    ids = [i["req_id"] for i in items]
    assert len(set(ids)) == len(ids), "duplicate req_ids"
    c.delete(f"/api/sessions/{sid}")


def test_truncation_collision_keeps_texts_unique(fuzz_app):
    c, _ = fuzz_app
    body = ("text\n" + "A" * 10_100 + " tail one\n" + "A" * 10_100 + " tail two\n").encode()
    check(c, body, "t.csv")


@given(st.binary(max_size=3000), st.sampled_from(NAMES))
@S
def test_random_bytes(fuzz_app, data, name):
    check(fuzz_app[0], data, name)


@st.composite
def csv_files(draw):
    delim = draw(st.sampled_from([",", ";", "\t", "|", ":", " "]))
    quoting = draw(st.sampled_from([csv.QUOTE_MINIMAL, csv.QUOTE_ALL, csv.QUOTE_NONNUMERIC]))
    nl = draw(st.sampled_from(["\n", "\r\n", "\r"]))
    header = draw(st.sampled_from(HEADERS))
    rows = draw(st.lists(st.lists(cell, min_size=1, max_size=len(header) + 2), max_size=40))
    buf = io.StringIO()
    w = csv.writer(buf, delimiter=delim, quoting=quoting, lineterminator=nl)
    w.writerow(header)
    w.writerows(rows)
    s = buf.getvalue()
    enc = draw(st.sampled_from(["utf-8", "utf-8-sig", "utf-16", "cp1252", "latin-1", "utf-32"]))
    return s.encode(enc, errors="replace")


@given(csv_files(), st.sampled_from(["a.csv", "A.CSV", "x.xls", "x.xlsx", "x.txt"]))
@S
def test_random_csv(fuzz_app, data, name):
    check(fuzz_app[0], data, name)


xl_val = st.one_of(st.none(), st.booleans(), st.integers(-10**12, 10**12), st.floats(allow_nan=False),
                   st.datetimes(min_value=datetime.datetime(1900, 1, 1)),
                   st.sampled_from(NASTY).filter(lambda s: "\x00" not in s and "\x01" not in s and "\x07" not in s),
                   st.text(alphabet=st.characters(blacklist_categories=("Cs", "Cc")), max_size=60))


@given(st.sampled_from(HEADERS), st.lists(st.lists(xl_val, min_size=1, max_size=6), max_size=40),
       st.integers(1, 3), st.sampled_from(["a.xlsx", "a.xls", "a.csv"]))
@S
def test_random_xlsx(fuzz_app, header, rows, nsheets, name):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(header)
    for r in rows:
        ws.append(r)
    for _ in range(nsheets - 1):
        wb.create_sheet().append(["text", "other"])
    buf = io.BytesIO()
    wb.save(buf)
    check(fuzz_app[0], buf.getvalue(), name)


@pytest.mark.parametrize("name,data", [
    ("big.csv", b"text\n" + b"The system shall do thing number %d ok\n" * 1 + b""),
    ("hdr.csv", b"id,text\n"), ("empty.csv", b""), ("bom.csv", b"\xef\xbb\xbf"), ("nums.csv", b"text\n1234567\n7654321\n"),
    ("trunc.xlsx", b"PK\x03\x04garbage"), ("zip.xlsx", b"PK\x05\x06" + b"\x00" * 18),
])
def test_edge_files(fuzz_app, name, data):
    check(fuzz_app[0], data, name)


def test_zip_bomb_xlsx_rejected(fuzz_app):
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/big.bin", b"\0" * (260 * 1024 * 1024))
    r = fuzz_app[0].post("/api/sessions", files={"file": ("b.xlsx", buf.getvalue())})
    assert r.status_code in (400, 413) and is_envelope(r)


def test_upload_limits(tmp_path):
    from .conftest import running_app
    with running_app(tmp_path, env={"REQCLUSTER_MAX_UPLOAD_MB": 1, "REQCLUSTER_MAX_ROWS": 50}) as (c, _):
        r = c.post("/api/sessions", files={"file": ("a.csv", b"text\n" + b"x" * (2 * 1024 * 1024))})
        assert r.status_code == 413 and is_envelope(r)
        rows = "".join(f"requirement number {i} shall hold\n" for i in range(80))
        r = c.post("/api/sessions", files={"file": ("a.csv", ("text\n" + rows).encode())})
        assert r.status_code == 413 and is_envelope(r)
        r = c.post("/api/sessions")  # no file
        assert 400 <= r.status_code < 500 and is_envelope(r)


def test_zz_fuzz_reached_success_path():
    # guards against a fuzzer that only ever produces 400s
    assert COUNT.get(200, 0) >= 40, COUNT
