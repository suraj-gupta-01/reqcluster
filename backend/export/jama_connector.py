"""Jama Connect export connector.

Builds an importable Jama item bundle (the shape accepted by Jama's REST
``POST /rest/v1/items`` and relationship endpoints). When a live Jama instance
is configured via ``JAMA_BASE_URL`` + ``JAMA_API_TOKEN``, items can be pushed;
otherwise the JSON bundle is returned for offline import.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

# Jama default item type / relationship type ids are instance-specific; these
# are overridable via env so the bundle matches a real project configuration.
DEFAULT_ITEM_TYPE = int(os.getenv("JAMA_ITEM_TYPE_ID", "33"))
DEFAULT_REL_TYPE = int(os.getenv("JAMA_RELATIONSHIP_TYPE_ID", "1"))


def build_jama_bundle(data: Dict[str, Any]) -> Dict[str, Any]:
    session = data.get("session", {})
    requirements: List[Dict[str, Any]] = data.get("requirements", [])
    clusters: List[Dict[str, Any]] = data.get("clusters", [])
    dependencies: List[Dict[str, Any]] = data.get("dependencies", [])
    project_id = int(os.getenv("JAMA_PROJECT_ID", "0")) or None
    cluster_label = {c["cluster_id"]: c.get("label", f"Cluster {c['cluster_id']}") for c in clusters}

    items: List[Dict[str, Any]] = []
    for r in requirements:
        rid = r.get("req_id") or f"REQ-{r.get('db_id')}"
        items.append({
            "documentKey": str(rid),
            "project": project_id,
            "itemType": DEFAULT_ITEM_TYPE,
            "fields": {
                "name": str(rid),
                "description": str(r.get("text", "")),
                "cluster": str(cluster_label.get(r.get("cluster_id"), "Noise")),
                "module": str(r.get("module") or ""),
            },
            "_ref": {"db_id": r.get("db_id")},
        })

    relationships: List[Dict[str, Any]] = []
    for edge in dependencies:
        relationships.append({
            "fromItem": str(edge.get("source_req_id")),
            "toItem": str(edge.get("target_req_id")),
            "relationshipType": DEFAULT_REL_TYPE,
            "_relation": edge.get("relation"),
        })

    return {
        "format": "jama",
        "project": project_id,
        "source": session.get("name"),
        "items": items,
        "relationships": relationships,
        "meta": {
            "item_count": len(items),
            "relationship_count": len(relationships),
            "note": "Set JAMA_PROJECT_ID/JAMA_ITEM_TYPE_ID to match the target Jama project.",
        },
    }


def export_jama(data: Dict[str, Any]) -> str:
    return json.dumps(build_jama_bundle(data), indent=2)
