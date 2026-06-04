"""SysML / UML XMI 2.5.1 exporter.

Emits a standards-compliant XMI 2.5.1 / UML 2.5 model importable by Papyrus and
MagicDraw. Mapping:

- the session            -> a root uml:Model
- each cluster           -> a uml:Package (a SysML requirement group)
- each requirement       -> a uml:Class tagged with the SysML "requirement"
                            keyword, the requirement id, and the text as an
                            owned comment (the SysML Requirement "text" property)
- each dependency edge   -> a uml:Dependency (client depends on supplier),
                            keyworded by relation type (deriveReqt / trace).

Full SysML profile stereotype application is tool-specific; this exporter emits
portable UML with SysML keywords, which the named tools import as a model tree.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

XMI_NS = "http://www.omg.org/spec/XMI/20131001"
UML_NS = "http://www.omg.org/spec/UML/20131001"


def _xmi(tag: str) -> str:
    return f"{{{XMI_NS}}}{tag}"


def _uml(tag: str) -> str:
    return f"{{{UML_NS}}}{tag}"


def _safe_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{value}"


def export_sysml_xmi(data: Dict[str, Any]) -> str:
    ET.register_namespace("xmi", XMI_NS)
    ET.register_namespace("uml", UML_NS)

    session = data.get("session", {})
    requirements: List[Dict[str, Any]] = data.get("requirements", [])
    clusters: List[Dict[str, Any]] = data.get("clusters", [])
    dependencies: List[Dict[str, Any]] = data.get("dependencies", [])
    model_name = str(session.get("name") or "ReqCluster Model")

    root = ET.Element(_xmi("XMI"))
    root.set(f"{{{XMI_NS}}}version", "2.5.1")

    model = ET.SubElement(root, _uml("Model"))
    model.set(_xmi("id"), "_model")
    model.set("name", model_name)

    db_id_to_class: Dict[Any, str] = {}
    reqs_by_cluster: Dict[Any, List[Dict[str, Any]]] = {}
    for r in requirements:
        reqs_by_cluster.setdefault(r.get("cluster_id"), []).append(r)

    def _add_requirement_class(parent: ET.Element, r: Dict[str, Any]) -> None:
        class_id = _safe_id("_req", r.get("db_id"))
        db_id_to_class[r.get("db_id")] = class_id
        rid = r.get("req_id") or f"REQ-{r.get('db_id')}"
        cls = ET.SubElement(parent, "packagedElement")
        cls.set(_xmi("type"), "uml:Class")
        cls.set(_xmi("id"), class_id)
        cls.set("name", f"«requirement» {rid}")
        # SysML Requirement text carried as an owned comment.
        comment = ET.SubElement(cls, "ownedComment")
        comment.set(_xmi("type"), "uml:Comment")
        comment.set(_xmi("id"), _safe_id("_txt", r.get("db_id")))
        body = ET.SubElement(comment, "body")
        body.text = str(r.get("text", ""))

    cluster_label = {c["cluster_id"]: c.get("label", f"Cluster {c['cluster_id']}") for c in clusters}

    # Clusters as packages.
    for c in clusters:
        cid = c["cluster_id"]
        pkg = ET.SubElement(model, "packagedElement")
        pkg.set(_xmi("type"), "uml:Package")
        pkg.set(_xmi("id"), _safe_id("_pkg", cid))
        pkg.set("name", str(cluster_label.get(cid, f"Cluster {cid}")))
        for r in reqs_by_cluster.get(cid, []):
            _add_requirement_class(pkg, r)

    # Noise / unclustered at model root.
    for r in reqs_by_cluster.get(-1, []) + reqs_by_cluster.get(None, []):
        _add_requirement_class(model, r)

    # Dependency edges as uml:Dependency (target depends on source).
    rel_keyword = {
        "data": "deriveReqt",
        "sequential": "trace",
        "hierarchical": "deriveReqt",
        "reference": "trace",
    }
    for i, edge in enumerate(dependencies):
        supplier = db_id_to_class.get(edge.get("source_db_id"))
        client = db_id_to_class.get(edge.get("target_db_id"))
        if not supplier or not client:
            continue
        dep = ET.SubElement(model, "packagedElement")
        dep.set(_xmi("type"), "uml:Dependency")
        dep.set(_xmi("id"), f"_dep_{i}")
        keyword = rel_keyword.get(edge.get("relation", "trace"), "trace")
        dep.set("name", f"«{keyword}»")
        dep.set("supplier", supplier)
        dep.set("client", client)

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
