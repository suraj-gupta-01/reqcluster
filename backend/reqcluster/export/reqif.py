"""ReqIF 1.2 exporter: clusters as heading spec-objects grouping requirements."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .bundle import ExportBundle, xml_id, xml_safe

REQIF_NS = "http://www.omg.org/spec/ReqIF/20110401/reqif.xsd"
XHTML_NS = "http://www.w3.org/1999/xhtml"


def _q(tag: str) -> str:
    return f"{{{REQIF_NS}}}{tag}"


def _el(parent, tag, attrs=None, text=None):
    el = ET.SubElement(parent, _q(tag))
    for k, v in (attrs or {}).items():
        el.set(k, xml_safe(v))
    if text is not None:
        el.text = xml_safe(text)
    return el


def export_reqif(b: ExportBundle) -> bytes:
    ET.register_namespace("", REQIF_NS)
    ET.register_namespace("xhtml", XHTML_NS)
    now, title = b.generated_at, b.session_name or "ReqCluster Export"
    seen: set = set()

    root = ET.Element(_q("REQ-IF"))
    rih = _el(_el(root, "THE-HEADER"), "REQ-IF-HEADER", {"IDENTIFIER": xml_id("_", "rc-header", seen)})
    _el(rih, "CREATION-TIME", text=now)
    _el(rih, "REQ-IF-TOOL-ID", text="ReqCluster")
    _el(rih, "REQ-IF-VERSION", text="1.0")
    _el(rih, "SOURCE-TOOL-ID", text="ReqCluster")
    _el(rih, "TITLE", text=title)
    rc = _el(_el(root, "CORE-CONTENT"), "REQ-IF-CONTENT")

    dts = _el(rc, "DATATYPES")
    _el(dts, "DATATYPE-DEFINITION-STRING", {"IDENTIFIER": "_dt-string", "LONG-NAME": "String",
                                            "LAST-CHANGE": now, "MAX-LENGTH": "32000"})
    _el(dts, "DATATYPE-DEFINITION-XHTML", {"IDENTIFIER": "_dt-xhtml", "LONG-NAME": "XHTML",
                                           "LAST-CHANGE": now})
    seen.update({"_dt-string", "_dt-xhtml"})

    types = _el(rc, "SPEC-TYPES")

    def attr_def(parent, kind, ident, name):
        ad = _el(parent, f"ATTRIBUTE-DEFINITION-{kind}",
                 {"IDENTIFIER": ident, "LONG-NAME": name, "LAST-CHANGE": now})
        _el(_el(ad, "TYPE"), f"DATATYPE-DEFINITION-{kind}-REF", text=f"_dt-{kind.lower()}")
        seen.add(ident)

    sot = _el(types, "SPEC-OBJECT-TYPE", {"IDENTIFIER": "_sot-req", "LONG-NAME": "Requirement",
                                          "LAST-CHANGE": now})
    sa = _el(sot, "SPEC-ATTRIBUTES")
    attr_def(sa, "STRING", "_ad-reqid", "ReqID")
    attr_def(sa, "XHTML", "_ad-text", "Text")
    attr_def(sa, "STRING", "_ad-cluster", "Cluster")
    soh = _el(types, "SPEC-OBJECT-TYPE", {"IDENTIFIER": "_sot-heading", "LONG-NAME": "Heading",
                                          "LAST-CHANGE": now})
    attr_def(_el(soh, "SPEC-ATTRIBUTES"), "STRING", "_ad-heading", "Heading")
    _el(types, "SPECIFICATION-TYPE", {"IDENTIFIER": "_st-spec",
                                      "LONG-NAME": "Requirements Specification", "LAST-CHANGE": now})
    seen.update({"_sot-req", "_sot-heading", "_st-spec"})

    def sval(vals, ref, value):
        av = _el(vals, "ATTRIBUTE-VALUE-STRING", {"THE-VALUE": value})
        _el(_el(av, "DEFINITION"), "ATTRIBUTE-DEFINITION-STRING-REF", text=ref)

    objs = _el(rc, "SPEC-OBJECTS")
    cluster_so = {}
    for c in b.clusters:
        cluster_so[c.id] = xml_id("_so-cluster-", c.id, seen)
        so = _el(objs, "SPEC-OBJECT", {"IDENTIFIER": cluster_so[c.id], "LAST-CHANGE": now})
        _el(_el(so, "TYPE"), "SPEC-OBJECT-TYPE-REF", text="_sot-heading")
        sval(_el(so, "VALUES"), "_ad-heading", c.label)

    req_so = {}
    for r in b.requirements:
        req_so[r.ordinal] = xml_id("_so-req-", r.ordinal, seen)
        so = _el(objs, "SPEC-OBJECT", {"IDENTIFIER": req_so[r.ordinal], "LAST-CHANGE": now})
        _el(_el(so, "TYPE"), "SPEC-OBJECT-TYPE-REF", text="_sot-req")
        vals = _el(so, "VALUES")
        sval(vals, "_ad-reqid", r.req_id)
        av = _el(vals, "ATTRIBUTE-VALUE-XHTML")
        _el(_el(av, "DEFINITION"), "ATTRIBUTE-DEFINITION-XHTML-REF", text="_ad-text")
        ET.SubElement(_el(av, "THE-VALUE"), f"{{{XHTML_NS}}}div").text = xml_safe(r.text)
        sval(vals, "_ad-cluster", r.cluster_label)

    specs = _el(rc, "SPECIFICATIONS")
    spec = _el(specs, "SPECIFICATION", {"IDENTIFIER": xml_id("_", "spec-1", seen),
                                        "LONG-NAME": title, "LAST-CHANGE": now})
    _el(_el(spec, "TYPE"), "SPECIFICATION-TYPE-REF", text="_st-spec")
    children = _el(spec, "CHILDREN")

    def hier(parent, so_id, key):
        sh = _el(parent, "SPEC-HIERARCHY", {"IDENTIFIER": xml_id("_sh-", key, seen), "LAST-CHANGE": now})
        _el(_el(sh, "OBJECT"), "SPEC-OBJECT-REF", text=so_id)
        return sh

    sub = {c.id: _el(hier(children, cluster_so[c.id], f"cluster-{c.id}"), "CHILDREN") for c in b.clusters}
    for r in b.requirements:
        hier(sub.get(r.cluster_id, children), req_so[r.ordinal], f"req-{r.ordinal}")

    ET.indent(root, space="  ")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")).encode()
