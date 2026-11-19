"""1. Upload boundary."""
import pytest

from .conftest import make_csv, make_xlsx, topic_texts
from .helpers import assert_json_error, csv_dangerous_cells

pytestmark = pytest.mark.acceptance

GOOD = topic_texts(3)


def test_uppercase_csv_extension_accepted(api):
    st, b = api.upload("REQS.CSV", make_csv(GOOD))
    assert st == 200, b


def test_uppercase_xlsx_extension_accepted(api):
    st, b = api.upload("REQS.XLSX", make_xlsx(GOOD))
    assert st == 200, b


def test_valid_xlsx_accepted(api):
    st, b = api.upload("reqs.xlsx", make_xlsx(GOOD))
    assert st == 200, b
    assert api.session(b["session_id"])[1]["total_requirements"] == len(GOOD)


def test_corrupt_xlsx_is_400_not_500(api):
    st, b = api.upload("bad.xlsx", b"this is definitely not a zip archive")
    assert_json_error(api, st, b, 400)


def test_truncated_xlsx_is_400(api):
    st, b = api.upload("bad.xlsx", make_xlsx(GOOD)[:200])
    assert_json_error(api, st, b, 400)


def test_xls_garbage_is_400(api):
    st, b = api.upload("old.xls", b"\xd0\xcf\x11\xe0 not really an xls")
    assert_json_error(api, st, b, 400)


def test_xls_with_xlsx_payload_handled_without_500(api):
    st, b = api.upload("old.xls", make_xlsx(GOOD))
    assert st in (200, 400), (st, b)


def test_unsupported_extension_400(api):
    st, b = api.upload("notes.txt", b"text\nhello there world\n")
    assert_json_error(api, st, b, 400)


def test_oversize_upload_413(api):
    st, b = api.upload("huge.csv", b"text\n" + b"a" * (30 * 1024 * 1024))
    assert_json_error(api, st, b, 413)


def test_empty_file_400(api):
    st, b = api.upload("empty.csv", b"")
    assert_json_error(api, st, b, 400)


def test_header_only_400(api):
    st, b = api.upload("h.csv", b"id,text\n")
    assert_json_error(api, st, b, 400)


def test_missing_text_column_400_with_message(api):
    st, b = api.upload("m.csv", b"id,foo\n1,bar baz qux quux\n2,bar baz qux corge\n")
    assert_json_error(api, st, b, 400)
    assert "text" in api.error_message(b).lower()


def test_bom_and_crlf_parse(api):
    raw = b"\xef\xbb\xbf" + make_csv(GOOD, newline="\r\n")
    st, b = api.upload("bom.csv", raw)
    assert st == 200, b
    reqs = api.all_requirements(b["session_id"])
    assert len(reqs) == len(GOOD)
    assert not any("\r" in r["text"] or "﻿" in r["text"] for r in reqs)
    assert reqs[0]["req_id"] == "REQ-001"


def test_quoted_commas_and_newlines_preserved(api):
    texts = [
        'The unit shall accept input A, B, and C without error',
        'The unit shall report "quoted" status,\nthen continue processing safely',
    ]
    sid = api.upload_texts(texts)
    got = [r["text"] for r in api.all_requirements(sid)]
    assert len(got) == 2
    assert got[0] == texts[0]
    assert got[1].replace("\n", " ").startswith('The unit shall report "quoted" status, then continue')


def test_cp1252_csv_decodes_correctly(api):
    text = "The pump ‘shall’ keep the € budget under 5 °C drift"
    raw = ("id,text\nR1," + text + "\nR2,The café controller shall remain idle\n").encode("cp1252")
    st, b = api.upload("win.csv", raw)
    assert st == 200, b
    got = [r["text"] for r in api.all_requirements(b["session_id"])]
    assert text in got
    assert "The café controller shall remain idle" in got


def test_duplicate_rows_collapsed(api):
    texts = ["The brake shall engage within 5 ms"] * 3 + ["The radio shall transmit on band A",
                                                          "The radio shall transmit on band B"]
    sid = api.upload_texts(texts)
    assert len(api.all_requirements(sid)) == 3
    assert api.session(sid)[1]["total_requirements"] == 3


def test_whitespace_only_rows_dropped(api):
    texts = ["   ", "\t\t", "The brake shall engage within 5 ms", "   ", "The radio shall transmit on band A"]
    sid = api.upload_texts(texts)
    got = [r["text"] for r in api.all_requirements(sid)]
    assert len(got) == 2 and all(t.strip() for t in got)


def test_csv_export_neutralises_formula_injection(api):
    texts = [
        "=HYPERLINK(\"http://evil\",\"x\") shall be rendered literally",
        "+1+1 shall be rendered literally in the sheet",
        "-2+3 shall be rendered literally in the sheet",
        "@SUM(A1:A9) shall be rendered literally in the sheet",
        "A normal requirement shall remain untouched here",
    ]
    ids = ["=cmd|' /C calc'!A0", "+7", "-9x", "@id", "OK-1"]
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "text"])
    for i, t in zip(ids, texts, strict=False):
        w.writerow([i, t])
    st, b = api.upload("inj.csv", buf.getvalue().encode())
    assert st == 200, b
    st, content, _ = api.export(b["session_id"], "csv")
    assert st == 200
    assert csv_dangerous_cells(content) == []
    assert b"shall be rendered literally" in content  # data kept, only neutralised


def test_feedback_csv_export_neutralises_formula_injection(clustered, api):
    sid = clustered
    rid = api.all_requirements(sid)[0]["id"]
    st, b = api.move(sid, rid, -1, comments="=1+1", applied_by="@evil")
    assert st == 200, b
    st, content = api.get("feedback_export", sid)
    assert st == 200
    assert csv_dangerous_cells(content) == []
