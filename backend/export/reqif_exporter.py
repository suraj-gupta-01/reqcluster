"""ReqIF 1.2 (OMG) exporter.

Produces a schema-shaped ReqIF XML document with a DATATYPES / SPEC-TYPES /
SPEC-OBJECTS / SPECIFICATIONS structure. Requirements become SPEC-OBJECTs
carrying ReqID, Text (XHTML), Module, and Cluster attributes; clusters become
heading SPEC-OBJECTs that group their requirements in a SPEC-HIERARCHY.

Targets import into ReqIF-aware tools (DOORS Next, ReqView, Polarion, RMF).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

REQIF_NS = "http://www.omg.org/spec/ReqIF/20110401/reqif.xsd"
XHTML_NS = "http://www.w3.org/1999/xhtml"


def _q(tag: str) -> str:
    return f"{{{REQIF_NS}}}{tag}"


def _xh(tag: str) -> str:
    return f"{{{XHTML_NS}}}{tag}"


def _el(parent: ET.Element, tag: str, attrs: Optional[Dict[str, str]] = None) -> ET.Element:
    el = ET.SubElement(parent, _q(tag))
    if attrs:
        for k, v in attrs.items():
            el.set(k, v)
    return el


def export_reqif(data: Dict[str, Any]) -> str:
    ET.register_namespace("", REQIF_NS)
    ET.register_namespace("xhtml", XHTML_NS)

    session = data.get("session", {})
    requirements: List[Dict[str, Any]] = data.get("requirements", [])
    clusters: List[Dict[str, Any]] = data.get("clusters", [])
    now = str(data.get("timestamp", "2025-01-01T00:00:00Z"))
    title = str(session.get("name") or "ReqCluster Export")

    root = ET.Element(_q("REQ-IF"))

    # --- Header ---
    header = _el(root, "THE-HEADER")
    rih = _el(header, "REQ-IF-HEADER", {"IDENTIFIER": "_rc-header"})
    _el(rih, "CREATION-TIME").text = now
    _el(rih, "REQ-IF-TOOL-ID").text = "ReqCluster"
    _el(rih, "REQ-IF-VERSION").text = "1.0"
    _el(rih, "SOURCE-TOOL-ID").text = "ReqCluster"
    _el(rih, "TITLE").text = title

    content = _el(root, "CORE-CONTENT")
    reqif_content = _el(content, "REQ-IF-CONTENT")

    # --- Datatypes ---
    datatypes = _el(reqif_content, "DATATYPES")
    _el(datatypes, "DATATYPE-DEFINITION-STRING", {
        "IDENTIFIER": "_dt-string", "LONG-NAME": "String",
        "LAST-CHANGE": now, "MAX-LENGTH": "32000",
    })
    _el(datatypes, "DATATYPE-DEFINITION-XHTML", {
        "IDENTIFIER": "_dt-xhtml", "LONG-NAME": "XHTML", "LAST-CHANGE": now,
    })

    # --- Spec types ---
    spec_types = _el(reqif_content, "SPEC-TYPES")
    sot = _el(spec_types, "SPEC-OBJECT-TYPE", {
        "IDENTIFIER": "_sot-req", "LONG-NAME": "Requirement", "LAST-CHANGE": now,
    })
    spec_attrs = _el(sot, "SPEC-ATTRIBUTES")

    def _attr_string(parent: ET.Element, identifier: str, name: str) -> None:
        ad = _el(parent, "ATTRIBUTE-DEFINITION-STRING", {
            "IDENTIFIER": identifier, "LONG-NAME": name, "LAST-CHANGE": now,
        })
        t = _el(ad, "TYPE")
        _el(t, "DATATYPE-DEFINITION-STRING-REF").text = "_dt-string"

    def _attr_xhtml(parent: ET.Element, identifier: str, name: str) -> None:
        ad = _el(parent, "ATTRIBUTE-DEFINITION-XHTML", {
            "IDENTIFIER": identifier, "LONG-NAME": name, "LAST-CHANGE": now,
        })
        t = _el(ad, "TYPE")
        _el(t, "DATATYPE-DEFINITION-XHTML-REF").text = "_dt-xhtml"

    _attr_string(spec_attrs, "_ad-reqid", "ReqID")
    _attr_xhtml(spec_attrs, "_ad-text", "Text")
    _attr_string(spec_attrs, "_ad-module", "Module")
    _attr_string(spec_attrs, "_ad-cluster", "Cluster")

    soth = _el(spec_types, "SPEC-OBJECT-TYPE", {
        "IDENTIFIER": "_sot-heading", "LONG-NAME": "Heading", "LAST-CHANGE": now,
    })
    soth_attrs = _el(soth, "SPEC-ATTRIBUTES")
    _attr_string(soth_attrs, "_ad-heading", "Heading")

    _el(spec_types, "SPECIFICATION-TYPE", {
        "IDENTIFIER": "_st-spec", "LONG-NAME": "Requirements Specification",
        "LAST-CHANGE": now,
    })

    # --- Spec objects ---
    spec_objects = _el(reqif_content, "SPEC-OBJECTS")
    cluster_label = {
        c["cluster_id"]: c.get("label", f"Cluster {c['cluster_id']}") for c in clusters
    }

    def _string_value(parent: ET.Element, ad_ref: str, value: str) -> None:
        av = _el(parent, "ATTRIBUTE-VALUE-STRING", {"THE-VALUE": value})
        d = _el(av, "DEFINITION")
        _el(d, "ATTRIBUTE-DEFINITION-STRING-REF").text = ad_ref

    def _xhtml_value(parent: ET.Element, ad_ref: str, value: str) -> None:
        av = _el(parent, "ATTRIBUTE-VALUE-XHTML")
        d = _el(av, "DEFINITION")
        _el(d, "ATTRIBUTE-DEFINITION-XHTML-REF").text = ad_ref
        tv = _el(av, "THE-VALUE")
        div = ET.SubElement(tv, _xh("div"))
        div.text = value

    for c in clusters:
        cid = c["cluster_id"]
        so = _el(spec_objects, "SPEC-OBJECT", {
            "IDENTIFIER": f"_so-cluster-{cid}", "LAST-CHANGE": now,
        })
        t = _el(so, "TYPE")
        _el(t, "SPEC-OBJECT-TYPE-REF").text = "_sot-heading"
        vals = _el(so, "VALUES")
        _string_value(vals, "_ad-heading", str(cluster_label.get(cid, f"Cluster {cid}")))

    for r in requirements:
        rid = r.get("req_id") or f"REQ-{r.get('db_id')}"
        so = _el(spec_objects, "SPEC-OBJECT", {
            "IDENTIFIER": f"_so-req-{r.get('db_id')}", "LAST-CHANGE": now,
        })
        t = _el(so, "TYPE")
        _el(t, "SPEC-OBJECT-TYPE-REF").text = "_sot-req"
        vals = _el(so, "VALUES")
        _string_value(vals, "_ad-reqid", str(rid))
        _xhtml_value(vals, "_ad-text", str(r.get("text", "")))
        _string_value(vals, "_ad-module", str(r.get("module") or ""))
        cid = r.get("cluster_id")
        _string_value(
            vals, "_ad-cluster",
            str(cluster_label.get(cid, "Noise" if cid == -1 else cid)),
        )

    # --- Specification hierarchy grouped by cluster ---
    specifications = _el(reqif_content, "SPECIFICATIONS")
    spec = _el(specifications, "SPECIFICATION", {
        "IDENTIFIER": "_spec-1", "LONG-NAME": title, "LAST-CHANGE": now,
    })
    st = _el(spec, "TYPE")
    _el(st, "SPECIFICATION-TYPE-REF").text = "_st-spec"
    children = _el(spec, "CHILDREN")

    reqs_by_cluster: Dict[Any, List[Dict[str, Any]]] = {}
    for r in requirements:
        reqs_by_cluster.setdefault(r.get("cluster_id"), []).append(r)

    for c in clusters:
        cid = c["cluster_id"]
        sh = _el(children, "SPEC-HIERARCHY", {
            "IDENTIFIER": f"_sh-cluster-{cid}", "LAST-CHANGE": now,
        })
        obj = _el(sh, "OBJECT")
        _el(obj, "SPEC-OBJECT-REF").text = f"_so-cluster-{cid}"
        sub_children = _el(sh, "CHILDREN")
        for r in reqs_by_cluster.get(cid, []):
            csh = _el(sub_children, "SPEC-HIERARCHY", {
                "IDENTIFIER": f"_sh-req-{r.get('db_id')}", "LAST-CHANGE": now,
            })
            cobj = _el(csh, "OBJECT")
            _el(cobj, "SPEC-OBJECT-REF").text = f"_so-req-{r.get('db_id')}"

    for r in reqs_by_cluster.get(-1, []) + reqs_by_cluster.get(None, []):
        sh = _el(children, "SPEC-HIERARCHY", {
            "IDENTIFIER": f"_sh-req-{r.get('db_id')}", "LAST-CHANGE": now,
        })
        obj = _el(sh, "OBJECT")
        _el(obj, "SPEC-OBJECT-REF").text = f"_so-req-{r.get('db_id')}"

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
