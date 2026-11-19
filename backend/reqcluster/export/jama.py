"""Jama Connect import bundle (JSON)."""
from __future__ import annotations

import json

from reqcluster.config import get_settings

from .bundle import ExportBundle


def export_jama(b: ExportBundle) -> bytes:
    st = get_settings()
    project = st.jama_project_id or None
    items = [{
        "documentKey": r.req_id,
        "project": project,
        "itemType": st.jama_item_type_id,
        "fields": {"name": r.req_id, "description": r.text, "cluster": r.cluster_label},
    } for r in b.requirements]
    rels = [{"fromItem": s, "toItem": t, "relationshipType": st.jama_rel_type_id, "_relation": typ}
            for s, t, typ, _w in b.edges]
    out = {"format": "jama", "project": project, "source": b.session_name, "items": items,
           "relationships": rels,
           "meta": {"item_count": len(items), "relationship_count": len(rels)}}
    return json.dumps(out, indent=2).encode("utf-8")
