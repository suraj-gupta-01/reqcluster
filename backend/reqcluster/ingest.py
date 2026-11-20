"""Upload ingest boundary: bytes -> cleaned, deduplicated RequirementIn rows."""
import csv
import io
import re
import zipfile
from dataclasses import dataclass, field

import openpyxl

from .domain import RequirementIn
from .errors import Invalid, TooLarge

MAX_TEXT_CHARS = 10_000
MAX_XLSX_UNZIPPED = 200 * 1024 * 1024
MIN_TEXT_CHARS = 6  # legacy rule: text length > 5
ID_NAMES = ["id", "req_id", "requirement_id", "req id", "identifier"]
TEXT_NAMES = ["text", "requirement", "description", "requirement text", "req text", "content", "shall"]
_TAGS = re.compile(r"</?[A-Za-z][A-Za-z0-9:-]*(\s[^<>]*)?/?>")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_WS = re.compile(r"\s+")


@dataclass
class IngestResult:
    requirements: list[RequirementIn]
    stats: dict
    warnings: list[str] = field(default_factory=list)


def read_capped(fileobj, max_bytes: int) -> bytes:
    buf = bytearray()
    while chunk := fileobj.read(1024 * 1024):
        buf += chunk
        if len(buf) > max_bytes:
            raise TooLarge(f"File exceeds the {max_bytes // (1024 * 1024)} MB upload limit.")
    return bytes(buf)


def _cell(value) -> str:
    """Ids, header names and attribute values: no control bytes (NUL breaks Postgres text)."""
    return _WS.sub(" ", _CTRL.sub("", str(value))).strip()


def _clean(text: str) -> str:
    text = _CTRL.sub("", _TAGS.sub(" ", text))
    return _WS.sub(" ", text).strip().strip("\"'").strip()


def _read_csv(data: bytes, bad: list) -> list[list[str]]:
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            s = data.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    else:  # latin-1 never fails; defensive
        raise Invalid("Could not decode the CSV file. Save it as UTF-8.")
    try:
        delim = csv.Sniffer().sniff(s[:65536], delimiters=",;\t|").delimiter
    except csv.Error:
        delim = ","
    rows = [r for r in csv.reader(io.StringIO(s, newline=""), delimiter=delim) if r]
    if not rows:
        raise ValueError("no header")
    width = len(rows[0])
    bad.extend(1 for r in rows[1:] if len(r) > width)
    return [rows[0]] + [r for r in rows[1:] if len(r) <= width]


def _read_xlsx(data: bytes) -> list[list[str]]:
    wb = openpyxl.load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        rows = [["" if c is None else str(c) for c in r] for r in wb.worksheets[0].iter_rows(values_only=True)]
    finally:
        wb.close()
    if not rows:
        raise ValueError("no header")
    return rows


def parse(data: bytes, filename: str, max_rows: int = 50000) -> IngestResult:
    if not data:
        raise Invalid("The uploaded file is empty.")
    name = (filename or "").lower()
    if name.endswith(".xls"):
        if not data.startswith(b"PK"):
            raise Invalid("Legacy .xls is not supported; save the file as .xlsx.")
        kind = "xlsx"
    elif name.endswith(".xlsx"):
        kind = "xlsx"
    elif name.endswith(".csv"):
        kind = "csv"
    else:
        raise Invalid("Unsupported file type. Allowed: .csv, .xlsx.")

    if kind == "xlsx":
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                total = sum(i.file_size for i in z.infolist())
        except zipfile.BadZipFile:
            raise Invalid("Could not read the XLSX file; it may be corrupt or malformed.") from None
        if total > MAX_XLSX_UNZIPPED:
            raise TooLarge("The spreadsheet expands to more than 200 MB; split it into smaller files.")

    bad: list = []
    try:
        table = _read_xlsx(data) if kind == "xlsx" else _read_csv(data, bad)
    except Invalid:
        raise
    except Exception:
        raise Invalid(f"Could not read the {kind.upper()} file; it may be corrupt or malformed.") from None

    header, body = table[0], table[1:]
    width = len(header)
    body = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in body]
    cols = {str(c).replace("\ufeff", "").strip().lower(): i for i, c in enumerate(header)}
    text_col = next((cols[n] for n in TEXT_NAMES if n in cols), None)
    if text_col is None:
        raise Invalid(
            f"No requirement text column found. Columns found: {', '.join(map(str, header)) or 'none'}. "
            f"Name one of: {', '.join(TEXT_NAMES[:4])}."
        )
    id_col = next((cols[n] for n in ID_NAMES if n in cols), None)
    extra = {i: _cell(str(c).replace("\ufeff", "")) for i, c in enumerate(header) if i not in (text_col, id_col)}
    extra = {i: n for i, n in extra.items() if n}

    stats = {"rows_read": len(body), "rows_kept": 0, "duplicates": 0, "empty_dropped": 0,
             "truncated": 0, "bad_lines": len(bad)}
    warnings: list[str] = []
    if bad:
        warnings.append(f"Skipped {len(bad)} malformed line(s).")

    seen_text: set[str] = set()
    seen_ids: dict[str, int] = {}
    used_ids: set[str] = set()
    out: list[RequirementIn] = []
    for row in body:
        raw_id = row[id_col] if id_col is not None else ""
        raw = row[text_col]
        ex = {i: row[i] for i in extra}
        text = _clean(str(raw))
        if len(text) < MIN_TEXT_CHARS:
            stats["empty_dropped"] += 1
            continue
        truncated = len(text) > MAX_TEXT_CHARS
        text = text[:MAX_TEXT_CHARS]  # before the dedupe key: stored texts must be unique
        key = text.casefold()
        if key in seen_text:
            stats["duplicates"] += 1
            continue
        seen_text.add(key)
        stats["truncated"] += truncated
        if len(out) >= max_rows:
            raise TooLarge(f"Too many requirements; the limit is {max_rows} rows.")
        ordinal = len(out) + 1
        rid = _cell(raw_id) or f"REQ-{ordinal:05d}"
        n = seen_ids.get(rid, 0)
        cand = rid
        while cand in used_ids:
            n += 1
            cand = f"{rid}-{n + 1}"
        seen_ids[rid] = n
        used_ids.add(cand)
        attrs = {extra[c]: v for c, v in ((c, _cell(v)) for c, v in ex.items()) if v}
        out.append(RequirementIn(req_id=cand, text=text, ordinal=ordinal, attributes=attrs))

    if not out:
        raise Invalid("No requirements found in the file (all rows were empty).")
    stats["rows_kept"] = len(out)
    if stats["truncated"]:
        warnings.append(f"{stats['truncated']} text(s) were truncated to {MAX_TEXT_CHARS} characters.")
    if len(out) < 5:
        warnings.append("Fewer than 5 requirements; clustering results will be limited.")
    return IngestResult(out, stats, warnings)
