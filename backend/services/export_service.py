"""Export service: assemble session data and dispatch to MBSE exporters."""

from __future__ import annotations

import csv
import io
from typing import Any, Dict, List

from sqlalchemy.orm import Session as DBSession

from export.jama_connector import export_jama
from export.pdf_report import export_pdf
from export.reqif_exporter import export_reqif
from export.sysml_xmi_exporter import export_sysml_xmi
from models.database import Cluster, DependencyTree, Requirement, Session, utcnow

SUPPORTED_FORMATS = {"reqif", "sysml", "jama", "csv", "pdf"}

_MEDIA = {
    "reqif": ("application/xml", "reqif"),
    "sysml": ("application/xml", "xmi"),
    "jama": ("application/json", "json"),
    "csv": ("text/csv", "csv"),
    "pdf": ("application/pdf", "pdf"),
}


class ExportServiceError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.message = message


def _gather(db: DBSession, session_id: int) -> Dict[str, Any]:
    session = db.query(Session).filter(Session.id == session_id).first()
    if not session:
        raise ExportServiceError(404, f"Session {session_id} not found.")
    reqs = (
        db.query(Requirement)
        .filter(Requirement.session_id == session_id)
        .order_by(Requirement.id.asc())
        .all()
    )
    if not reqs:
        raise ExportServiceError(400, "No requirements found for session.")
    clusters = (
        db.query(Cluster)
        .filter(Cluster.session_id == session_id)
        .order_by(Cluster.cluster_id.asc())
        .all()
    )

    requirements = [
        {
            "db_id": r.id,
            "req_id": r.req_id,
            "text": r.text,
            "module": r.module,
            "section": r.section,
            "cluster_id": r.cluster_id,
            "is_noise": r.is_noise,
        }
        for r in reqs
    ]
    cluster_dicts = [
        {"cluster_id": c.cluster_id, "label": c.label, "keywords": c.keywords, "size": c.size}
        for c in clusters
    ]

    # Dependency edges (if generated) mapped to requirement DB ids for tracing.
    dependencies: List[Dict[str, Any]] = []
    dep = db.query(DependencyTree).filter(DependencyTree.session_id == session_id).first()
    if dep and dep.nodes and dep.edges:
        index_to_db = {n["id"]: None for n in dep.nodes}
        # dependency nodes use the ordered index; map back to requirement db ids.
        for idx, r in enumerate(reqs):
            if idx in index_to_db:
                index_to_db[idx] = r.id
        node_req_id = {n["id"]: n["node_id"] for n in dep.nodes}
        for e in dep.edges:
            dependencies.append({
                "source_db_id": index_to_db.get(e["source"]),
                "target_db_id": index_to_db.get(e["target"]),
                "source_req_id": node_req_id.get(e["source"]),
                "target_req_id": node_req_id.get(e["target"]),
                "relation": e.get("relation"),
            })

    return {
        "session": {"id": session.id, "name": session.name, "filename": session.filename},
        "requirements": requirements,
        "clusters": cluster_dicts,
        "dependencies": dependencies,
        "timestamp": utcnow().isoformat() + "Z",
    }


def _export_csv(data: Dict[str, Any]) -> str:
    cluster_label = {c["cluster_id"]: c["label"] for c in data["clusters"]}
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["req_id", "text", "module", "section", "cluster_id", "cluster_label", "is_noise"])
    for r in data["requirements"]:
        writer.writerow([
            r.get("req_id"), r.get("text"), r.get("module"), r.get("section"),
            r.get("cluster_id"),
            cluster_label.get(r.get("cluster_id"), "Noise" if r.get("cluster_id") == -1 else ""),
            r.get("is_noise"),
        ])
    return out.getvalue()


def export_session(db: DBSession, session_id: int, fmt: str):
    """Return (content, media_type, filename) for the requested export format.

    `content` is str for text formats and bytes for PDF (FastAPI's Response
    accepts both).
    """
    fmt = (fmt or "").strip().lower()
    if fmt not in SUPPORTED_FORMATS:
        raise ExportServiceError(
            400, f"Unsupported format '{fmt}'. Use one of: {', '.join(sorted(SUPPORTED_FORMATS))}."
        )
    data = _gather(db, session_id)

    if fmt == "reqif":
        content = export_reqif(data)
    elif fmt == "sysml":
        content = export_sysml_xmi(data)
    elif fmt == "jama":
        content = export_jama(data)
    elif fmt == "pdf":
        content = export_pdf(data)
    else:
        content = _export_csv(data)

    media_type, ext = _MEDIA[fmt]
    filename = f"reqcluster_session_{session_id}.{ext}"
    return content, media_type, filename
