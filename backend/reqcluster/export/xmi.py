"""SysML / UML XMI 2.5.1 exporter: clusters -> packages, requirements -> classes."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .bundle import ExportBundle, xml_id, xml_safe

XMI_NS = "http://www.omg.org/spec/XMI/20131001"
UML_NS = "http://www.omg.org/spec/UML/20131001"
_KEYWORD = {"data": "deriveReqt", "hierarchical": "deriveReqt"}


def _xmi(tag: str) -> str:
    return f"{{{XMI_NS}}}{tag}"


def _set(el, **attrs):
    for k, v in attrs.items():
        el.set(k, xml_safe(v))


def export_sysml(b: ExportBundle) -> bytes:
    ET.register_namespace("xmi", XMI_NS)
    ET.register_namespace("uml", UML_NS)
    seen: set = {"_model"}
    root = ET.Element(_xmi("XMI"))
    root.set(_xmi("version"), "2.5.1")
    model = ET.SubElement(root, f"{{{UML_NS}}}Model")
    model.set(_xmi("id"), "_model")
    _set(model, name=b.session_name or "ReqCluster Model")

    parents = {}
    for c in b.clusters:
        pkg = ET.SubElement(model, "packagedElement")
        pkg.set(_xmi("type"), "uml:Package")
        pkg.set(_xmi("id"), xml_id("_pkg_", c.id, seen))
        _set(pkg, name=c.label)
        parents[c.id] = pkg

    class_of: dict[str, str] = {}
    for r in b.requirements:
        cls = ET.SubElement(parents.get(r.cluster_id, model), "packagedElement")
        cid = xml_id("_req_", r.ordinal, seen)
        class_of.setdefault(r.req_id, cid)
        cls.set(_xmi("type"), "uml:Class")
        cls.set(_xmi("id"), cid)
        _set(cls, name=f"«requirement» {r.req_id}")
        cm = ET.SubElement(cls, "ownedComment")
        cm.set(_xmi("type"), "uml:Comment")
        cm.set(_xmi("id"), xml_id("_txt_", r.ordinal, seen))
        ET.SubElement(cm, "body").text = xml_safe(r.text)

    for i, (src, tgt, typ, _w) in enumerate(b.edges):
        if src not in class_of or tgt not in class_of:
            continue
        dep = ET.SubElement(model, "packagedElement")
        dep.set(_xmi("type"), "uml:Dependency")
        dep.set(_xmi("id"), xml_id("_dep_", i, seen))
        _set(dep, name=f"«{_KEYWORD.get(typ, 'trace')}»", supplier=class_of[src], client=class_of[tgt])

    ET.indent(root, space="  ")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")).encode()
