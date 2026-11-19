"""Normalised export model shared by every exporter."""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import datetime, UTC

from sqlalchemy import select

from reqcluster.db import session_scope
from reqcluster.domain import NOISE
from reqcluster.errors import NotFound
from reqcluster.models import AssignmentRow, ClusterRow, RequirementRow, SessionRow

_XML_BAD = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff￾￿]")
_NCNAME_BAD = re.compile(r"[^A-Za-z0-9_.\-]")
_NUMBER = re.compile(r"-?\d+(\.\d+)?")


@dataclass
class ExportReq:
    req_id: str
    text: str
    cluster_id: int | None  # None = unclustered, -1 = noise
    cluster_label: str
    is_noise: bool
    ordinal: int


@dataclass
class ExportCluster:
    id: int
    label: str
    keywords: list
    size: int


@dataclass
class ExportBundle:
    session_id: int
    session_name: str
    requirements: list[ExportReq]
    clusters: list[ExportCluster]
    generated_at: str
    # Extension point: (source_req_id, target_req_id, type, weight); filled once dependencies exist.
    edges: list = field(default_factory=list)


def build(sid: int) -> ExportBundle:
    with session_scope() as s:
        row = s.get(SessionRow, sid)
        if row is None:
            raise NotFound(f"session {sid} not found")
        run_id = row.current_run_id
        labels: dict[int, str] = {}
        clusters: list[ExportCluster] = []
        assign: dict[int, int] = {}
        if run_id is not None:
            for c in s.scalars(select(ClusterRow).where(ClusterRow.run_id == run_id)
                               .order_by(ClusterRow.cluster_id)):
                labels[c.cluster_id] = c.label
                if c.cluster_id != NOISE:
                    clusters.append(ExportCluster(c.cluster_id, c.label, list(c.keywords or []), c.size))
            assign = dict(s.execute(select(AssignmentRow.requirement_id, AssignmentRow.cluster_id)
                                    .where(AssignmentRow.run_id == run_id)).all())
        reqs = []
        for r in s.scalars(select(RequirementRow).where(RequirementRow.session_id == sid)
                           .order_by(RequirementRow.ordinal)):
            cid = assign.get(r.id)
            if cid is None:
                label = "Unclustered"
            elif cid == NOISE:
                label = "Noise"
            else:
                label = labels.get(cid, f"Cluster {cid}")
            reqs.append(ExportReq((r.req_id or "").strip() or f"REQ-{r.ordinal:05d}", r.text,
                                  cid, label, cid == NOISE, r.ordinal))
        return ExportBundle(sid, row.name, reqs, clusters,
                            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))


def xml_safe(s) -> str:
    """Strip characters illegal in XML 1.0."""
    return _XML_BAD.sub("", str(s if s is not None else ""))


def xml_id(prefix: str, value, seen: set) -> str:
    """Unique NCName-safe id; `prefix` must itself start with a letter or underscore."""
    base = prefix + _NCNAME_BAD.sub("_", str(value))
    out, n = base, 1
    while out in seen:
        n += 1
        out = f"{base}_{n}"
    seen.add(out)
    return out


def _guard(v) -> str:
    v = "" if v is None else str(v)
    if v and v[0] in "=+-@\t\r" and not _NUMBER.fullmatch(v):
        return "'" + v
    return v


def to_csv(b: ExportBundle) -> bytes:
    buf = io.StringIO(newline="")
    w = csv.writer(buf)
    w.writerow(["req_id", "text", "cluster_id", "cluster_label", "is_noise", "ordinal"])
    for r in b.requirements:
        w.writerow([_guard(x) for x in (r.req_id, r.text, "" if r.cluster_id is None else r.cluster_id,
                                         r.cluster_label, r.is_noise, r.ordinal)])
    return buf.getvalue().encode("utf-8-sig")
