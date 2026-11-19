import io

import pytest
from openpyxl import Workbook

from reqcluster.errors import Invalid, TooLarge
from reqcluster.ingest import parse, read_capped

A, B, C = "The brake shall engage within 5 ms", "The radio shall transmit on band A", "The pump shall keep 5 bar"


def csv_bytes(s: str, enc="utf-8") -> bytes:
    return s.encode(enc)


def xlsx(rows) -> bytes:
    wb = Workbook()
    for r in rows:
        wb.active.append(r)
    b = io.BytesIO()
    wb.save(b)
    return b.getvalue()


def texts(res):
    return [r.text for r in res.requirements]


def test_cp1252_and_latin1():
    r = parse(csv_bytes("id,text\nR1,The ‘pump’ shall cost 5 €\nR2,The café shall idle\n", "cp1252"), "a.csv")
    assert texts(r)[0] == "The ‘pump’ shall cost 5 €"
    r = parse(csv_bytes("text\nThe café shall idle\n", "latin-1"), "a.csv")
    assert "café" in texts(r)[0]


def test_bom_crlf():
    r = parse(b"\xef\xbb\xbfid,text\r\nR1," + A.encode() + b"\r\n", "a.csv")
    assert texts(r) == [A] and r.requirements[0].req_id == "R1"


@pytest.mark.parametrize("d", [";", "\t", "|"])
def test_delimiters(d):
    r = parse(f"id{d}text\nR1{d}{A}\nR2{d}{B}\n".encode(), "a.csv")
    assert texts(r) == [A, B]


def test_xlsx_upper_and_xls_with_xlsx_payload():
    data = xlsx([["id", "text"], ["R1", A]])
    assert texts(parse(data, "A.XLSX")) == [A]
    assert texts(parse(data, "a.xls")) == [A]


def test_xls_garbage():
    with pytest.raises(Invalid, match="xls"):
        parse(b"\xd0\xcf\x11\xe0 junk", "a.xls")


def test_corrupt_and_truncated_xlsx():
    with pytest.raises(Invalid):
        parse(b"not a zip", "a.xlsx")
    with pytest.raises(Invalid):
        parse(xlsx([["text"], [A]])[:200], "a.xlsx")


def test_bad_extension_empty_header_only_missing_col():
    for data, name in [(b"x", "a.txt"), (b"", "a.csv"), (b"id,text\n", "a.csv"), (b"text\n  \n\t\n", "a.csv")]:
        with pytest.raises(Invalid):
            parse(data, name)
    with pytest.raises(Invalid, match="id, foo"):
        parse(b"id,foo\n1,bar baz qux\n", "a.csv")


def test_duplicate_ids_and_texts():
    r = parse(f"id,text\nR1,{A}\nR1,{B}\nR1,{C}\nR2,{A.upper()}\n".encode(), "a.csv")
    assert [x.req_id for x in r.requirements] == ["R1", "R1-2", "R1-3"]
    assert r.stats["duplicates"] == 1 and r.stats["rows_kept"] == 3


def test_generated_ids_and_few_rows_warning():
    r = parse(f"id,text\n,{A}\n2,   \n,{B}\n".encode(), "a.csv")
    assert [x.req_id for x in r.requirements] == ["REQ-00001", "REQ-00002"]
    assert r.stats["empty_dropped"] == 1 and r.warnings


def test_truncation():
    r = parse(("text\n" + "a" * 20000 + "\n").encode(), "a.csv")
    assert len(texts(r)[0]) == 10_000 and r.stats["truncated"] == 1


def test_too_many_rows():
    body = "text\n" + "\n".join(f"Requirement number {i} shall hold" for i in range(4))
    with pytest.raises(TooLarge):
        parse(body.encode(), "a.csv", max_rows=3)


def test_read_capped():
    assert read_capped(io.BytesIO(b"abc"), 3) == b"abc"
    with pytest.raises(TooLarge):
        read_capped(io.BytesIO(b"abcd"), 3)


def test_quoted_comma_and_newline():
    s = 'text\n"A, B, and C shall work"\n"The unit shall say ""hi"",\nthen go"\n'
    r = parse(s.encode(), "a.csv")
    assert texts(r) == ["A, B, and C shall work", 'The unit shall say "hi", then go']


def test_zip_bomb_rejected():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("x.xlsx", b"\0" * (201 * 1024 * 1024))
    with pytest.raises(TooLarge):
        parse(buf.getvalue(), "a.xlsx")


def test_suffixed_ids_stay_unique():
    data = csv_bytes(f"id,text\nR1,{A}\nR1,{B}\nR1-2,{C}\n")
    ids = [r.req_id for r in parse(data, "a.csv").requirements]
    assert len(set(ids)) == 3


def test_ids_headers_and_attribute_values_are_cleaned_like_text():
    # NUL/control bytes anywhere in the file must not reach the DB (Postgres rejects NUL in text)
    data = b"id,text,mod\x00ule\nR\x001,The pump shall start within 5 seconds,The\x07rmal\n"
    r = parse(data, "a.csv").requirements[0]
    blob = r.req_id + "".join(k + v for k, v in r.attributes.items())
    assert "\x00" not in blob and "\x07" not in blob
    assert r.req_id == "R1" and r.attributes == {"module": "Thermal"}
