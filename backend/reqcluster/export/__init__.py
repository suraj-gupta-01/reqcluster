"""Export entry point: export(sid, fmt) -> (bytes, media_type, filename)."""
from __future__ import annotations

from reqcluster.errors import Invalid

from .bundle import build, to_csv
from .jama import export_jama
from .pdf import export_pdf
from .reqif import export_reqif
from .xmi import export_sysml

_FORMATS = {
    "reqif": (export_reqif, "application/xml", "reqif"),
    "sysml": (export_sysml, "application/xml", "xmi"),
    "jama": (export_jama, "application/json", "json"),
    "csv": (to_csv, "text/csv", "csv"),
    "pdf": (export_pdf, "application/pdf", "pdf"),
}


def export(sid: int, fmt: str) -> tuple[bytes, str, str]:
    if fmt not in _FORMATS:
        raise Invalid(f"unknown export format {fmt!r}; expected one of {', '.join(_FORMATS)}")
    fn, media, ext = _FORMATS[fmt]
    return fn(build(sid)), media, f"session-{sid}.{ext}"
