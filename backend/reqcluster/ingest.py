"""Upload ingest boundary: bytes -> cleaned, deduplicated RequirementIn rows."""
import csv
import io
import re
from dataclasses import dataclass, field

import pandas as pd

from .domain import RequirementIn
from .errors import Invalid, TooLarge

MAX_TEXT_CHARS = 10_000
MIN_TEXT_CHARS = 6  # legacy rule: text length > 5
ID_NAMES = ["id", "req_id", "requirement_id", "req id", "identifier"]
TEXT_NAMES = ["text", "requirement", "description", "requirement text", "req text", "content", "shall"]
_TAGS = re.compile(r"<[^>]+>")
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


def _clean(text: str) -> str:
    text = _CTRL.sub("", _TAGS.sub(" ", text))
    return _WS.sub(" ", text).strip().strip("\"'").strip()


def _read_csv(data: bytes, bad: list) -> pd.DataFrame:
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
    return pd.read_csv(
        io.StringIO(s), sep=delim, dtype=str, keep_default_na=False,
        engine="python", on_bad_lines=lambda row: bad.append(1),
    )


def _read_xlsx(data: bytes) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(data), sheet_name=0, dtype=str, engine="openpyxl").fillna("")


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

    bad: list = []
    try:
        df = _read_xlsx(data) if kind == "xlsx" else _read_csv(data, bad)
    except Invalid:
        raise
    except Exception:
        raise Invalid(f"Could not read the {kind.upper()} file; it may be corrupt or malformed.") from None

    cols = {str(c).replace("﻿", "").strip().lower(): c for c in df.columns}
    text_col = next((cols[n] for n in TEXT_NAMES if n in cols), None)
    if text_col is None:
        raise Invalid(
            f"No requirement text column found. Columns found: {', '.join(map(str, df.columns)) or 'none'}. "
            f"Name one of: {', '.join(TEXT_NAMES[:4])}."
        )
    id_col = next((cols[n] for n in ID_NAMES if n in cols), None)

    stats = {"rows_read": len(df), "rows_kept": 0, "duplicates": 0, "empty_dropped": 0,
             "truncated": 0, "bad_lines": len(bad)}
    warnings: list[str] = []
    if bad:
        warnings.append(f"Skipped {len(bad)} malformed line(s).")

    seen_text: set[str] = set()
    seen_ids: dict[str, int] = {}
    out: list[RequirementIn] = []
    ids = df[id_col].tolist() if id_col is not None else [""] * len(df)
    for raw_id, raw in zip(ids, df[text_col].tolist(), strict=True):
        text = _clean(str(raw))
        if len(text) < MIN_TEXT_CHARS:
            stats["empty_dropped"] += 1
            continue
        key = text.casefold()
        if key in seen_text:
            stats["duplicates"] += 1
            continue
        seen_text.add(key)
        if len(text) > MAX_TEXT_CHARS:
            text = text[:MAX_TEXT_CHARS]
            stats["truncated"] += 1
        if len(out) >= max_rows:
            raise TooLarge(f"Too many requirements; the limit is {max_rows} rows.")
        ordinal = len(out) + 1
        rid = _WS.sub(" ", str(raw_id)).strip() or f"REQ-{ordinal:05d}"
        n = seen_ids.get(rid, 0) + 1
        seen_ids[rid] = n
        out.append(RequirementIn(req_id=rid if n == 1 else f"{rid}-{n}", text=text, ordinal=ordinal))

    if not out:
        raise Invalid("No requirements found in the file (all rows were empty).")
    stats["rows_kept"] = len(out)
    if stats["truncated"]:
        warnings.append(f"{stats['truncated']} text(s) were truncated to {MAX_TEXT_CHARS} characters.")
    if len(out) < 5:
        warnings.append("Fewer than 5 requirements; clustering results will be limited.")
    return IngestResult(out, stats, warnings)
